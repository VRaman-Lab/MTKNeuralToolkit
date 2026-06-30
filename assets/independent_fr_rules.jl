# =============================================================================
# Custom frule/rrule Through MTK ODE Sensitivity Pipeline
# =============================================================
# Demonstrates custom ChainRulesCore frule and rrule flowing through ODE
# sensitivity computation for an MTK-compiled system.
#
# Working:
#   - Forward mode: frule via ForwardDiff.Dual dispatch                    ✅
#   - Mooncake direct: rrule via @from_rrule bridge                        ✅
#   - Mixed mode: Mooncake outer + ForwardDiffSensitivity inner (frule)     ✅
#
# Blocked by ecosystem bugs or my incompetence :
#   - Reverse mode: Mooncake + InterpolatingAdjoint/MooncakeVJP (rrule)
#     Bug 1: user_set_discontinuity field missing (SciMLBase ↔ OrdinaryDiffEq)
#     Bug 2: Mooncake stack overflow on MTKParameters/ODEProblem tangent types
#
# Future: ForwardDiffSensitivity with convert_tspan=true supports callbacks/events.
# =============================================================================

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using DifferentiationInterface
import ForwardDiff
import Mooncake
using ChainRulesCore
using SciMLStructures: Tunable, canonicalize, replace
using SymbolicIndexingInterface: parameter_values, setp
using PreallocationTools: DiffCache, get_tmp
using SciMLSensitivity

# =============================================================================
# 1. PRIMAL FUNCTION & MTK REGISTRATION
# =============================================================================
# (x >= 0.0) * 1.0 works with Float64, Num (symbolic), and ForwardDiff.Dual
function custom_surrogate(x)
    return (x >= 0.0) * 1.0
end

@register_symbolic custom_surrogate(x)

# =============================================================================
# 2. CUSTOM FRULE & RRULE (ChainRulesCore)
# =============================================================================
const FAKE_SLOPE_FWD = 42.0   # forward-mode surrogate derivative
const FAKE_SLOPE_REV = -99.0  # reverse-mode surrogate derivative

function ChainRulesCore.frule((Δself, Δx), ::typeof(custom_surrogate), x)
    y = custom_surrogate(x)
    return y, FAKE_SLOPE_FWD * Δx
end

function ChainRulesCore.rrule(::typeof(custom_surrogate), x)
    y = custom_surrogate(x)
    function custom_surrogate_pullback(Δy)
        return NoTangent(), Δy * FAKE_SLOPE_REV
    end
    return y, custom_surrogate_pullback
end

# =============================================================================
# 2b. FORWARD DIFF DUAL DISPATCH (frule → ForwardDiff)
# =============================================================================
# ForwardDiff doesn't consult ChainRulesCore.frule — it propagates Dual numbers
# through raw code. This dispatch manually invokes the frule to compute the
# directional derivative, bridging the two systems.
function custom_surrogate(d::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    primal = ForwardDiff.value(d)
    partials = ForwardDiff.partials(d)
    y, dy = ChainRulesCore.frule((NoTangent(), one(V)), custom_surrogate, primal)
    newvals = ntuple(i -> dy * partials[i], N)
    return ForwardDiff.Dual{T}(y, ForwardDiff.Partials(newvals))
end

# =============================================================================
# 2c. MOONCAKE BRIDGE (CRC rrule → Mooncake rrule!!)
# =============================================================================
# Mooncake doesn't auto-bridge CRC rrules. @from_rrule creates a Mooncake
# rrule!! that wraps the existing ChainRulesCore.rrule.
# For both forward+reverse in Mooncake, use @from_chainrules instead.
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(custom_surrogate), Float64}

# =============================================================================
# 3. BUILD MTK SYSTEM
# =============================================================================
println("--- Building Minimal MTK System ---")
@parameters p_val
@variables v(t)
eqs = [D(v) ~ custom_surrogate(p_val)]
@mtkcompile sys = System(eqs, t)
u0_and_p = [v => -0.5, p_val => 2.0]
prob = ODEProblem(sys, u0_and_p, (0.0, 1.0))

# =============================================================================
# 4. PREPARE CONSTANTS (per MTK optimization docs)
# =============================================================================
# Store problem + helpers as a tuple, pass via DI.Constant to avoid globals
# (Mooncake's __verify_const fails on Num globals from @variables/@parameters).
# DiffCache provides a type-stable preallocated buffer (works with Duals).
# setp provides an efficient setter for specific parameters.
setter = setp(prob, [p_val])
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(prob))[1]))
save_points = [1.0]
CONSTS = (prob, save_points, setter, diffcache)

# =============================================================================
# 5. LOSS FUNCTIONS (following MTK docs pattern)
# =============================================================================
# All loss functions take (x, p) where x = optimization variables, p = constants.
# Uses DiffCache for type-stable preallocation and setp for efficient parameter
# setting. Uses numeric indexing (sol.u[end][1]) instead of symbolic (sol[v][end])
# to avoid Mooncake __verify_const errors on Num globals.
#
# NOTE: For Mooncake, use last(sol.u)[1] instead of sol[end][1] — the latter
# has a known BoundsError bug in SciMLBaseMooncakeExt._scatter_pullback.

function _update_prob(x, p)
    odeprob = p[1]; setter = p[3]; dc = p[4]
    ps = parameter_values(odeprob)
    buffer = get_tmp(dc, x)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps = replace(Tunable(), ps, buffer)
    setter(ps, x)
    return remake(odeprob; p = ps)
end

# Forward mode: ForwardDiff ignores sensealg; Dual dispatch handles custom_surrogate.
function loss_fwd(x, p)
    newprob = _update_prob(x, p)
    sol = solve(newprob, Tsit5(); saveat = p[2])
    return sol.u[end][1]
end

# Mixed mode: Mooncake outer + ForwardDiffSensitivity inner.
# ForwardDiffSensitivity propagates ForwardDiff Duals through the solver,
# hitting our Dual dispatch → frule. Mooncake uses SciMLSensitivity's
# rrule for solve (which internally calls ForwardDiff), so Mooncake never
# needs to differentiate through the solver itself.
# Event-compatible with convert_tspan=true (for future spiking neuron models).
function loss_mixed(x, p)
    newprob = _update_prob(x, p)
    sol = solve(newprob, Tsit5(); saveat = p[2],
        sensealg = SciMLSensitivity.ForwardDiffSensitivity())
    return sol.u[end][1]
end

# Reverse mode: Mooncake + InterpolatingAdjoint/MooncakeVJP.
# MooncakeVJP differentiates the RHS → hits @from_rrule-bridged CRC rrule.
# BLOCKED by two ecosystem bugs:
#   1. user_set_discontinuity field missing (SciMLBase ↔ OrdinaryDiffEqCore)
#   2. Mooncake stack overflow on MTKParameters/ODEProblem tangent types
# File bug reports; will work when both are fixed.
function loss_rev(x, p)
    newprob = _update_prob(x, p)
    sol = solve(newprob, Tsit5(); saveat = p[2],
        sensealg = SciMLSensitivity.InterpolatingAdjoint(
            autojacvec = SciMLSensitivity.MooncakeVJP()))
    return sol.u[end][1]
end

# =============================================================================
# 6. RUN TESTS
# =============================================================================
initial_guess = [2.0]
moon_backend = AutoMooncake(; config = Mooncake.Config(; friendly_tangents = true))

# Expected results:
# ODE: D(v) = custom_surrogate(p_val), v(0) = -0.5
# Primal: v(1.0) = -0.5 + custom_surrogate(2.0) * 1.0 = 0.5
# Gradient = slope * ∫(0→1) dt = slope * 1.0
#   Forward (frule, slope=42.0):   [42.0]
#   Reverse (rrule, slope=-99.0): [-99.0]

println("\n=== Test 0: Direct Mooncake on custom_surrogate (rrule bridge) ===")
g_direct(x) = custom_surrogate(x[1])
grad0 = gradient(g_direct, moon_backend, [2.0])
println("Result:   ", grad0, "  Expected: [-99.0]  ", grad0 ≈ [-99.0] ? "✅" : "❌")

println("\n=== Test 1: Forward-Mode (ForwardDiff + Dual dispatch → frule) ===")
fwd_backend = AutoForwardDiff()
fwd_prep = prepare_gradient(loss_fwd, fwd_backend, initial_guess, Constant(CONSTS))
fwd_grad = gradient(loss_fwd, fwd_prep, fwd_backend, initial_guess, Constant(CONSTS))
println("Result:   ", fwd_grad, "  Expected: [42.0]  ", fwd_grad ≈ [42.0] ? "✅" : "❌")

println("\n=== Test 2: Mixed-Mode (Mooncake + ForwardDiffSensitivity → frule) === (commented out as results in seg fault)")
# try
#     mixed_prep = prepare_gradient(loss_mixed, moon_backend, initial_guess, Constant(CONSTS))
#     mixed_grad = gradient(loss_mixed, mixed_prep, moon_backend, initial_guess, Constant(CONSTS))
#     println("Result:   ", mixed_grad, "  Expected: [42.0]  ", mixed_grad ≈ [42.0] ? "✅" : "❌")
# catch e
#     println("FAILED: ", sprint(showerror, e))
# end

println("\n=== Test 3: Reverse-Mode (Mooncake + InterpolatingAdjoint/MooncakeVJP → rrule) ===")
println("  (Blocked by ecosystem bugs — file reports at SciML/Mooncake GitHub)")
try
    rev_prep = prepare_gradient(loss_rev, moon_backend, initial_guess, Constant(CONSTS))
    rev_grad = gradient(loss_rev, rev_prep, moon_backend, initial_guess, Constant(CONSTS))
    println("Result:   ", rev_grad, "  Expected: [-99.0]  ", rev_grad ≈ [-99.0] ? "✅" : "❌")
catch e
    println("FAILED: ", sprint(showerror, e))
end

println("""

=== Summary ===
Custom derivative mechanism:
  frule:  CRC frule + ForwardDiff.Dual dispatch (forward mode)
  rrule:  CRC rrule + Mooncake.@from_rrule (reverse mode)

Test 0 — rrule bridge:        $(grad0 ≈ [-99.0] ? "✅" else "❌")  Mooncake uses CRC rrule directly
Test 1 — forward mode:       $(fwd_grad ≈ [42.0] ? "✅" else "❌")  ForwardDiff + Dual dispatch → frule
Test 2 — mixed mode:          see above  Mooncake + ForwardDiffSensitivity → frule
Test 3 — reverse mode:       see above  Blocked by ecosystem bugs (file reports)

Bug reports to file:
  1. https://github.com/SciML/SciMLSensitivity.jl/issues
     FieldError: ODEIntegrator has no field 'user_set_discontinuity'
  2. https://github.com/chalk-lab/Mooncake.jl/issues
     Stack overflow computing tangent types for MTK ODEProblem/MTKParameters

Event support:
  ForwardDiffSensitivity with convert_tspan=true is compatible with callbacks.
  Add convert_tspan=true to the sensealg when adding events:
    sensealg = ForwardDiffSensitivity(convert_tspan=true)
""")

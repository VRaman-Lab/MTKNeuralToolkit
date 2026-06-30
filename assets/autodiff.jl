using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using DifferentiationInterface
import ForwardDiff
import Zygote
using ChainRulesCore
using SciMLStructures: Tunable, replace
using SciMLSensitivity
import SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP, ZygoteAdjoint

# ==========================================
# 1. PRIMAL FUNCTION & MTK REGISTRATION
# ==========================================
function custom_surrogate(x)
    return x >= 0.0 ? 1.0 : 0.0
end

@register_symbolic custom_surrogate(x)

# ==========================================
# 2. CUSTOM FRULE & RRULE
# ==========================================
const FAKE_SLOPE_FWD = 42.0
const FAKE_SLOPE_REV = -99.0

function ChainRulesCore.frule((Δself, Δx), ::typeof(custom_surrogate), x)
    y = custom_surrogate(x)
    ∂y = FAKE_SLOPE_FWD * Δx
    return y, ∂y
end

function ChainRulesCore.rrule(::typeof(custom_surrogate), x)
    y = custom_surrogate(x)
    function custom_surrogate_pullback(Δy)
        return NoTangent(), Δy * FAKE_SLOPE_REV
    end
    return y, custom_surrogate_pullback
end

# ==========================================
# 2b. FORWARD DIFF DUAL DISPATCH (frule bridge)
# ==========================================
# ForwardDiff does NOT consult ChainRulesCore.frule. It propagates Dual
# numbers through raw code. We bridge this by dispatching on Dual and
# manually invoking the frule to compute the directional derivative.
# This is the standard pattern for custom functions in ForwardDiff pipelines.
function custom_surrogate(d::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    primal = ForwardDiff.value(d)
    partials = ForwardDiff.partials(d)
    y, dy = ChainRulesCore.frule((NoTangent(), one(V)), custom_surrogate, primal)
    newvals = ntuple(i -> dy * partials[i], N)
    return ForwardDiff.Dual{T}(y, ForwardDiff.Partials(newvals))
end

# ==========================================
# 3. BUILD MTK SYSTEM
# ==========================================
println("--- Building Minimal MTK System ---")
@parameters p_val
@variables v(t)
eqs = [D(v) ~ custom_surrogate(p_val)]
@mtkcompile sys = System(eqs, t)
u0_and_p = [v => -0.5, p_val => 2.0]
prob = ODEProblem(sys, u0_and_p, (0.0, 1.0))

# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================
# Forward-mode: ForwardDiff ignores sensealg. Our Dual dispatch method
# is called when Duals flow through the RHS during the solver's internal
# operations.
function loss_fwd(p_wrapper)
    ps = replace(Tunable(), prob.p, p_wrapper)
    new_prob = remake(prob; p = ps)
    sol = solve(new_prob, Tsit5(); saveat = 1.0)
    return sol[v][end]
end

# Reverse-mode with continuous adjoint + Zygote VJP:
# The adjoint computes VJPs of the RHS using Zygote, which differentiates
# through the compiled MTK function and hits our custom rrule.
function loss_rev_adjoint(p_wrapper)
    ps = replace(Tunable(), prob.p, p_wrapper)
    new_prob = remake(prob; p = ps)
    sol = solve(new_prob, Tsit5(); saveat = 1.0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))
    return sol[v][end]
end

# Reverse-mode with full Zygote (discretize-then-optimize):
# Zygote differentiates through the entire ODE solver. Every RHS evaluation
# is differentiated by Zygote, hitting our custom rrule.
function loss_rev_full(p_wrapper)
    ps = replace(Tunable(), prob.p, p_wrapper)
    new_prob = remake(prob; p = ps)
    sol = solve(new_prob, Tsit5(); saveat = 1.0,
        sensealg = ZygoteAdjoint())
    return sol[v][end]
end

# ==========================================
# 5. RUN TESTS
# ==========================================
initial_guess = [2.0]

# Expected results:
# ODE:  D(v) = custom_surrogate(p_val),  v(0) = -0.5
# Primal: v(1.0) = -0.5 + custom_surrogate(2.0) * 1.0 = 0.5
# Gradient = slope * ∫(0→1) dt = slope * 1.0
#   Forward (frule slope = 42.0):  [42.0]
#   Reverse (rrule slope = -99.0): [-99.0]

println("\n=== Forward-Mode AD (ForwardDiff + Dual dispatch → frule) ===")
fwd_backend = AutoForwardDiff()
fwd_prep = prepare_gradient(loss_fwd, fwd_backend, initial_guess)
fwd_grad = gradient(loss_fwd, fwd_prep, fwd_backend, initial_guess)
println("Result:   ", fwd_grad)
println("Expected: [42.0]")
println("Match:    ", fwd_grad ≈ [42.0])

println("\n=== Reverse-Mode AD (Zygote + InterpolatingAdjoint/ZygoteVJP → rrule) ===")
rev_backend = AutoZygote()
rev_prep = prepare_gradient(loss_rev_adjoint, rev_backend, initial_guess)
rev_grad = gradient(loss_rev_adjoint, rev_prep, rev_backend, initial_guess)
println("Result:   ", rev_grad)
println("Expected: [-99.0]")
println("Match:    ", rev_grad ≈ [-99.0])

println("\n=== Reverse-Mode AD (Zygote + ZygoteAdjoint → rrule) ===")
rev_prep2 = prepare_gradient(loss_rev_full, rev_backend, initial_guess)
rev_grad2 = gradient(loss_rev_full, rev_prep2, rev_backend, initial_guess)
println("Result:   ", rev_grad2)
println("Expected: [-99.0]")
println("Match:    ", rev_grad2 ≈ [-99.0])

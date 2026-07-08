using SafeTestsets
using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named, @variables, @parameters, t_nounits as t, D_nounits as D, extend, @component, @unpack, System
using ModelingToolkitStandardLibrary.Electrical: OnePort
using OrdinaryDiffEq
using Test

# ------------------------------------------------------------------------------
# 1. Shared Gating Dynamics (from Example 1)
# ------------------------------------------------------------------------------
hh_na_m = v -> (
    0.1 .* (v .+ 40.0) ./ (1.0 .- exp.(-(v .+ 40.0) ./ 10.0)),
    4.0 .* exp.(-(v .+ 65.0) ./ 18.0)
)
hh_na_h = v -> (
    0.07 .* exp.(-(v .+ 65.0) ./ 20.0),
    1.0 ./ (1.0 .+ exp.(-(v .+ 35.0) ./ 10.0))
)

n_alpha(v) = 0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0))
n_beta(v)  = 0.125 .* exp.(-(v .+ 65.0) ./ 80.0)
n_inf(v)   = n_alpha(v) ./ (n_alpha(v) .+ n_beta(v))
tau_n(v)   = 1.0 ./ (n_alpha(v) .+ n_beta(v))
hh_k_n_inftau = MTKNeuralToolkit.InfTau(n_inf, tau_n)

sodium_gates = [
    GateSpec(:m, 3, 0.052, hh_na_m), 
    GateSpec(:h, 1, 0.596, hh_na_h)
]

potassium_gates = [
    GateSpec(:n_gate, 4, 0.317, hh_k_n_inftau)
]

top = Scalar()

@testset "Standard Hodgkin-Huxley Single Compartment" begin
    @named soma_cap = Capacitor(topology=top, C=1.0)
    @named na_ch = GenericChannel(topology=top, g=120.0, E_rev=50.0,  gates=sodium_gates)
    @named k_ch  = GenericChannel(topology=top, g=36.0,  E_rev=-77.0, gates=potassium_gates)
    @named leak  = GenericChannel(topology=top, g=0.3,   E_rev=-54.4, gates=GateSpec[])

    channels = [na_ch, k_ch, leak]
    soma = build_compartment(soma_cap, channels; name=:soma, V_init=-65.0, topology=top)

    drivers = [(1, 10.0)]
    net = build_acausal_network([soma]; drivers=drivers, name=:single_neuron)

    sys = mtkcompile(net.sys)
    prob = ODEProblem(sys, [], (0.0, 100.0))
    sol = solve(prob, Rosenbrock23(), reltol=1e-4, abstol=1e-4)

    @test sol.retcode == ReturnCode.Success
    V = sol[sys.soma.soma_cap.v]
    @test all(!isnan, V)
    # Check that it spikes: max voltage should be > 0 mV
    @test maximum(V) > 0.0
    # Check that it starts from the initial condition
    @test V[1] ≈ -65.0
end

# ------------------------------------------------------------------------------
# 2. Custom Components from First Principles (from Example 1b)
# ------------------------------------------------------------------------------
@component function CustomLeakChannel(; name, g=0.3, E_rev=-54.4)
    @named oneport = OnePort()
    @unpack v, i = oneport
    @parameters g=g E_rev=E_rev
    eqs = [i ~ g * (v - E_rev)]
    return extend(System(eqs, t, [], [g, E_rev]; name=name), oneport)
end

@component function CustomNaPChannel(; name, g=10.0, E_rev=50.0, V_init=-65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    @parameters g=g E_rev=E_rev
    @variables m(t) = 1.0 / (1.0 + exp(-(V_init + 50.0) / 5.0))
    
    m_inf(V) = 1.0 / (1.0 + exp(-(V + 50.0) / 5.0))
    tau_m = 5.0
    
    eqs = [
        D(m) ~ (m_inf(v) - m) / tau_m,
        i ~ g * m * (v - E_rev)
    ]
    return extend(System(eqs, t, [m], [g, E_rev]; name=name), oneport)
end

@testset "Custom Ion Channels from First Principles" begin
    @named cap = Capacitor(topology=top, C=1.0)
    @named leak = CustomLeakChannel()
    @named nap = CustomNaPChannel()

    channels = [leak, nap]
    soma = build_compartment(cap, channels; name=:soma_custom, V_init=-65.0, topology=top)

    drivers = [(1, 2.0)]
    net = build_acausal_network([soma]; drivers=drivers, name=:custom_neuron)

    sys = mtkcompile(net.sys)

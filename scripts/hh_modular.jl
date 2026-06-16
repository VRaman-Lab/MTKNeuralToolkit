using ModelingToolkit
import ModelingToolkitStandardLibrary.Electrical: Ground, OnePort
using ModelingToolkitStandardLibrary.Blocks: RealInput
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkit: mtkcompile
using OrdinaryDiffEq
using Plots



"""
Soma Component: Represents a pure physical lipid bilayer membrane patch.
"""
@component function Capacitor(; name, C = 1.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters C = C
    vars = @variables V(t) = -65.0
    
    eqs = [
        D(v) ~ i / C
        V ~ v
    ]
    extend(System(eqs, t, vars, [C]; name), oneport)
end

"""
CurrentSource Component: Converts a causal RealInput signal (u) 
into an acausal electrical current (i) injecting into a physical Node.
"""
@component function CurrentSource(; name)
    @named oneport = OnePort()
    @unpack i = oneport
    @named I = RealInput()
    
    eqs = [
        i ~ -I.u 
    ]
    extend(System(eqs, t, [], []; name, systems = [I]), oneport)
end

"""
fixed_reversal Component: A pure constant voltage source (Nernst battery).
"""
@component function fixed_reversal(; name, E = 0.0)
    @named oneport = OnePort()
    @unpack v = oneport
    params = @parameters E = E
    eqs = [v ~ E]
    extend(System(eqs, t, [], [E]; name), oneport)
end

"""
nagates Component: Pure Sodium channel gating resistance. 
"""
@component function nagates(; name, g = 120.0) 
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters g = g
    vars = @variables begin
        m_gate(t)=0.0
        h_gate(t)=1.0
        αₘ(t)
        αₕ(t)
        βₘ(t)
        βₕ(t)
    end
    eqs = [
        αₘ ~ 0.182 * (v + 35.0) / (1.0 - exp(-(v + 35.0) / 9.0))
        βₘ ~ -0.124 * (v + 35.0) / (1.0 - exp((v + 35.0) / 9.0))
        αₕ ~ 0.25 * exp(-(v + 90.0) / 12.0) 
        βₕ ~ 0.25 * (exp((v + 62.0) / 6.0)) / exp(-(v + 90.0) / 12.0) 
        D(m_gate) ~ αₘ * (1 - m_gate) - βₘ * m_gate
        D(h_gate) ~ αₕ * (1 - h_gate) - βₕ * h_gate
        i ~ g * m_gate^3 * h_gate * v 
    ]
    extend(System(eqs, t, vars, [g]; name), oneport)
end

"""
kgates Component: Pure Potassium channel gating resistance.
"""
@component function kgates(; name, g = 36.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters g = g
    vars = @variables begin
        n_gate(t) = 0.0
        αₙ(t)
        βₙ(t)
    end
    eqs = [
        αₙ ~ 0.02 * (v - 25.0) / (1.0 - exp(-(v - 25.0) / 9.0))
        βₙ ~ -0.002 * (v - 25.0) / (1.0 - exp((v - 25.0) / 9.0))
        D(n_gate) ~ αₙ * (1 - n_gate) - βₙ * n_gate
        i ~ v * n_gate^4 * g
    ]
    extend(System(eqs, t, vars, [g]; name), oneport)
end

"""
lgates Component: Pure passive Leak channel resistance.
"""
@component function lgates(; name, g = 0.3)
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters g = g
    eqs = [i ~ v * g]
    extend(System(eqs, t, [], [g]; name), oneport)
end


"""
build_channel: Factory function that wires a gating mechanism in series 
with an ionic reversal potential battery.
"""
function build_channel(gate, battery; name)
    eqs = [
        connect(gate.n, battery.p)
    ]
    return System(eqs, t, [], []; name, systems = [gate, battery])
end

"""
build_neuron: Builder function that automatically compiles a parallel connection matrix
across a Soma and a hardcoded internal CurrentSource injector.
"""
function build_neuron(soma, channels; stimulus_block=nothing, name=:neuron)
    @named ground = Ground()
    @named injector = CurrentSource() # Built-in injector source
    
    eqs = [
        connect(soma.n, ground.g)
        
        # Parallel networks hook up the channels AND the built-in injector branch
        connect(soma.p, [ch.gate.p for ch in channels]..., injector.p)
        connect(soma.n, [ch.batt.n for ch in channels]..., injector.n)
    ]
    
    all_systems = [soma, ground, injector, channels...]
    
    # Structural evaluation to ensure equation balancing
    if stimulus_block !== nothing
        # If an external block is passed, connect it to the inner injector's RealInput
        push!(eqs, connect(stimulus_block.output, injector.I))
        push!(all_systems, stimulus_block)
    else
        # If no block is passed, pin the injector to 0.0 to balance the system matrix
        push!(eqs, injector.I.u ~ 0.0)
    end
    
    return System(eqs, t, [], []; name, systems = all_systems)
end


# =============================================================================
# Runtime
# =============================================================================

@named soma = Capacitor(C = 1.0)
@named stimulus_block = Blocks.Sine(frequency = 0.1, amplitude = 10.0)

sodium    = build_channel(nagates(name=:gate), fixed_reversal(E = 50.0, name=:batt); name=:sodium)
potassium = build_channel(kgates(name=:gate), fixed_reversal(E = -77.0, name=:batt); name=:potassium)
leak      = build_channel(lgates(name=:gate), fixed_reversal(E = -54.4, name=:batt); name=:leak)

hh_neuron = build_neuron(soma, [sodium, potassium, leak]; stimulus_block = stimulus_block, name = :hh_neuron)

hh_compiled = mtkcompile(hh_neuron)
prob = ODEProblem(hh_compiled, [], (0.0, 50.0))
sol = solve(prob, Rosenbrock23())

plot(sol, idxs=[soma.V], title="Hardcoded Injector Architecture Spike", xlabel="Time", ylabel="Voltage (mV)")
plot(sol, idxs = [potassium.gate.v])

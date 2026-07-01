# ==========================================
# Synapse Components
# ==========================================

"""
    SynapsePort

A boundary connector that exposes the postsynaptic current variable (`I_syn`) 
and binds it to the positive pin (`p.i`) of a standard electrical port. 

This component is typically used internally by compartment builders to route 
synaptic currents into a postsynaptic compartment's `CurrentSource`.
"""
@component function SynapsePort(; name, topology=Scalar())
    if topology isa Scalar
        @named p = Pin()
        @variables I_syn(t)
        vars = SymbolicT[I_syn]
        eqs = Equation[p.i ~ I_syn]
    else
        @named p = VectorizedPin(N=topology.N)
        @variables I_syn(t)[1:topology.N]
        vars = SymbolicT[I_syn]
        eqs = Equation[p.i ~ I_syn]
    end
    return System(eqs, t, vars, SymbolicT[]; systems=[p], name=name)
end

"""
    CholSynapse(; name, g_max=30.0, E_rev=-80.0, k_minus=0.01, V_th=-35.0, delta=5.0, geometry=NoGeometry())

A continuous cholinergic synapse model. The synaptic state variable `s` represents 
the fraction of open receptors. It rises towards a steady-state `s_inf` governed by 
the presynaptic voltage, and decays exponentially.

The synaptic current is calculated as the current injected into the postsynaptic membrane:
`I_syn = g_max * s * (E_rev - V_post)`

# Arguments
- `g_max`: Maximum synaptic conductance (scaled by geometry if provided).
- `E_rev`: Reversal potential of the synapse (e.g., -80 mV for inhibitory).
- `k_minus`: Rate constant for receptor unbinding (controls decay time).
- `V_th`: Half-activation voltage for the presynaptic sigmoid.
- `delta`: Slope of the presynaptic sigmoid activation.
- `geometry`: AbstractGeometry struct for scaling `g_max`.
"""
@component function CholSynapse(; name, g_max=30.0, E_rev=-80.0, k_minus=0.01, V_th=-35.0, delta=5.0, geometry=NoGeometry())
    g_max_val = get_synaptic_conductance(g_max, geometry)
    
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max_val E_rev=E_rev k_minus=k_minus V_th=V_th delta=delta
    
    s_inf = 1.0 / (1.0 + exp((V_th - V_pre) / delta))
    tau_s = (1.0 - s_inf) / k_minus
    
    eqs = Equation[
        D(s) ~ (s_inf - s) / tau_s,
        I_syn ~ g_max * s * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, E_rev, k_minus, V_th, delta]; systems=System[], name=name)
end

"""
    GlutSynapse(; name, g_max=30.0, E_rev=-70.0, k_minus=0.025, V_th=-35.0, delta=5.0, geometry=NoGeometry())

A continuous glutamatergic synapse model. Behaves identically to `CholSynapse` but uses 
default parameters typical for fast excitatory glutamatergic receptors.

# Arguments
- `g_max`: Maximum synaptic conductance (scaled by geometry if provided).
- `E_rev`: Reversal potential of the synapse (e.g., -70 mV or higher for excitatory).
- `k_minus`: Rate constant for receptor unbinding.
- `V_th`: Half-activation voltage for the presynaptic sigmoid.
- `delta`: Slope of the presynaptic sigmoid activation.
- `geometry`: AbstractGeometry struct for scaling `g_max`.
"""
@component function GlutSynapse(; name, g_max=30.0, E_rev=-70.0, k_minus=0.025, V_th=-35.0, delta=5.0, geometry=NoGeometry())
    g_max_val = get_synaptic_conductance(g_max, geometry)
    
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max_val E_rev=E_rev k_minus=k_minus V_th=V_th delta=delta
    
    s_inf = 1.0 / (1.0 + exp((V_th - V_pre) / delta))
    tau_s = (1.0 - s_inf) / k_minus
    
    eqs = Equation[
        D(s) ~ (s_inf - s) / tau_s,
        I_syn ~ g_max * s * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, E_rev, k_minus, V_th, delta]; systems=System[], name=name)
end

"""
    ExpSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

A simple exponential decay synapse. The synaptic gating variable `s` is driven by a 
continuous sigmoidal function of the presynaptic voltage and decays exponentially with time constant `τ`.

The current injected into the postsynaptic compartment is:
`I_syn = g_max * s * (E_rev - V_post)`

# Arguments
- `g_max`: Maximum synaptic conductance.
- `τ`: Decay time constant of the synapse.
- `E_rev`: Reversal potential of the synapse.
- `V_th`: Threshold voltage for presynaptic activation.
- `slope`: Slope of the presynaptic sigmoid activation.
"""
@component function ExpSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope]; 
                  systems=System[], name=name)
end

"""
    AlphaSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

An alpha-function synapse implemented via a cascade of two first-order filters (`s1` and `s2`). 
This produces the classic unimodal alpha-function response in synaptic conductance following 
a sustained presynaptic depolarization.

The current injected into the postsynaptic compartment is:
`I_syn = g_max * s2 * (E_rev - V_post)`

# Arguments
- `g_max`: Maximum synaptic conductance.
- `τ`: Time constant for both cascaded filters.
- `E_rev`: Reversal potential of the synapse.
- `V_th`: Threshold voltage for presynaptic activation.
- `slope`: Slope of the presynaptic sigmoid activation.
"""
@component function AlphaSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s1(t)=0.0 s2(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s1) ~ -s1 / τ + σ(V_pre - V_th),
        D(s2) ~ -s2 / τ + s1,
        I_syn ~ g_max * s2 * (E_rev - V_post)
    ]
    return System(eqs, t, [s1, s2, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, slope]; systems=System[], name=name)
end

"""
    NMDASynapse(; name, g_max=1.0, τ=100.0, E_rev=0.0, V_th=-20.0, Mg_conc=1.0, slope=2.0)

An N-Methyl-D-Aspartate (NMDA) receptor synapse. It includes the classic voltage-dependent 
Magnesium block that reduces conductance at hyperpolarized potentials. The gating variable `s` 
decays with a slow time constant `τ`.

The current injected into the postsynaptic compartment is:
`I_syn = g_max * s * mg_block(V_post) * (E_rev - V_post)`

# Arguments
- `g_max`: Maximum synaptic conductance.
- `τ`: Slow decay time constant typical of NMDA receptors.
- `E_rev`: Reversal potential of the synapse (usually near 0 mV).
- `V_th`: Threshold voltage for presynaptic activation.
- `Mg_conc`: Extracellular Magnesium concentration determining block strength.
- `slope`: Slope of the presynaptic sigmoid activation.
"""
@component function NMDASynapse(; name, g_max=1.0, τ=100.0, E_rev=0.0, V_th=-20.0, 
                                  Mg_conc=1.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th Mg_conc=Mg_conc slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    mg_block(V) = 1.0 / (1.0 + Mg_conc * exp(-0.062 * V))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * mg_block(V_post) * (E_rev - V_post)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, Mg_conc, slope]; systems=System[], name=name)
end

"""
    VectorizedExpSynapse(; name, N_pre, N_post, W, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

A vectorized block of exponential synapses representing a dense `N_post` by `N_pre` projection. 
It accepts an entire weight matrix `W` mapping presynaptic gating variables to postsynaptic currents.

The synaptic state `s` is a vector of length `N_pre`. The postsynaptic current vector is computed via:
`I_syn[i] = g_max * sum_j(W[i, j] * s[j]) * (E_rev - V_post[i])`

# Arguments
- `N_pre`: Number of presynaptic elements.
- `N_post`: Number of postsynaptic elements.
- `W`: A matrix of connection weights (dimensions `N_post` x `N_pre`).
- `g_max`: Maximum global synaptic conductance.
- `τ`: Decay time constant.
- `E_rev`: Reversal potential of the synapse.
- `V_th`: Threshold voltage for presynaptic activation.
- `slope`: Slope of the presynaptic sigmoid activation.
"""
@component function VectorizedExpSynapse(; name, N_pre, N_post, W,
                                            g_max=1.0, τ=5.0, E_rev=0.0,
                                            V_th=-20.0, slope=2.0)
    @variables s(t)[1:N_pre] I_syn(t)[1:N_post] V_pre(t)[1:N_pre] V_post(t)[1:N_post]
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    # Make W a symbolic parameter!
    @parameters W[1:N_post, 1:N_pre]=W

    σ(V) = 1.0 ./ (1.0 .+ exp.(-(V .- V_th) ./ slope))
    synaptic_drive = W * s
    
    eqs = [
        D(s) ~ -s ./ τ .+ σ(V_pre),
        I_syn ~ g_max .* (E_rev .- V_post) .* synaptic_drive
    ]
    
    init_conds = Dict(s => zeros(N_pre))
    
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope, W];
                  systems=System[], 
                  initial_conditions=init_conds, 
                  name=name)
end

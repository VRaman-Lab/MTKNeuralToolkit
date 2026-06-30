This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: src/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
src/
  components/
    calcium.jl
    channels.jl
    electrical.jl
    synapses.jl
  library/
    ContinuousSpikers.jl
    HodgkinHuxley.jl
  connections.jl
  MTKNeuralToolkit.jl
  network.jl
  topology.jl
```

# Files

## File: src/library/ContinuousSpikers.jl
```julia
module ContinuousSpikers
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, Scalar, Vectorized, OnePort
    using ModelingToolkit: t_nounits as t, D_nounits as D, @named, @variables, @parameters, @component, System, Equation, SymbolicT, extend, @unpack
    using Symbolics: unwrap

    # ==========================================
    # 1. Morris-Lecar (Built via GenericChannel)
    # ==========================================
    
    # Fast Ca2+ gating (effectively instantaneous)
    const V1, V2 = -20.0, 15.0
    const ml_ca_m = v -> (
        0.5 .* (1.0 .+ tanh.((v .- V1) ./ V2)) ./ 0.1,
        0.5 .* (1.0 .- tanh.((v .- V1) ./ V2)) ./ 0.1
    )

    # Slow K+ gating (recovery variable)
    const V3, V4 = -25.0, 5.0
    const tau_n = 10.0
    const ml_k_n = v -> (
        0.5 .* (1.0 .+ tanh.((v .- V3) ./ V4)) ./ tau_n,
        0.5 .* (1.0 .- tanh.((v .- V3) ./ V4)) ./ tau_n
    )

    @component function MorrisLecar(; name, topology=Scalar(), V_init=-20.0, 
                          g_Ca=4.0, E_Ca=100.0, g_K=8.5, E_K=-70.0, g_L=0.1, E_L=-50.0)
        m0 = 0.5 * (1 + tanh((V_init - V1) / V2))
        n0 = 0.5 * (1 + tanh((V_init - V3) / V4))
        
        ca_gates = [GateSpec(:m, 1, m0, ml_ca_m)]
        k_gates  = [GateSpec(:n, 1, n0, ml_k_n)]
        
        # Note: In a real build script, you'd create the Capacitor separately, 
        # but for convenience we can just document the required channels.
        # We return a tuple of the channels to be used with build_compartment.
        @named ca_ch = GenericChannel(topology=topology, g=g_Ca, E_rev=E_Ca, gates=ca_gates)
        @named k_ch  = GenericChannel(topology=topology, g=g_K, E_rev=E_K, gates=k_gates)
        @named leak  = GenericChannel(topology=topology, g=g_L, E_rev=E_L, gates=GateSpec[])
        
        return (ca_ch, k_ch, leak)
    end

    # ==========================================
    # 2. FitzHugh-Nagumo (Custom 2D OnePort)
    # ==========================================
    
    @component function FitzHughNagumo(; name, topology=Scalar(), I_ext=0.0, a=0.7, b=0.8, c=10.0, tau=12.5)
        if topology isa Scalar
            @named oneport = OnePort()
            @unpack v, i = oneport
            
            @parameters a=a b=b c=c tau=tau
            params = SymbolicT[a, b, c, tau]
            
            @variables w(t)=0.0
            vars = SymbolicT[v, w]
            
            # The channel provides the cubic and recovery dynamics.
            # C * dV/dt = I_ext - i_channel
            # We want: dV/dt = c * (v - v^3/3 - w) + I_ext
            # So: i_channel = -c * (v - v^3/3 - w)
            eqs = Equation[
                i ~ -c * (v - (v^3)/3.0 - w),
                D(w) ~ (v + a - b * w) / tau
            ]
            
            return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        else
            N = topology.N
            @named oneport = VectorizedOnePort(N=N)
            @unpack v, i = oneport
            
            @parameters a=a b=b c=c tau=tau
            params = SymbolicT[a, b, c, tau]
            
            @variables w(t)[1:N]=zeros(N)
            vars = SymbolicT[v, w]
            
            eqs = Equation[
                i ~ -c .* (v .- (v.^3)./ 3.0 .- w),
                D(w) ~ (v .+ a .- b .* w) ./ tau
            ]
            
            return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        end
    end

    export MorrisLecar, FitzHughNagumo
end
```

## File: src/components/calcium.jl
```julia
@connector function CaPort(; name, topology=Scalar())
    if topology isa Scalar
        vars = @variables begin
            Ca(t)
            J_Ca(t), [connect = Flow]
        end
    else
        vars = @variables begin
            Ca(t)[1:topology.N]
            J_Ca(t)[1:topology.N], [connect = Flow]
        end
    end
    return System(Equation[], t, vars, SymbolicT[]; name=name)
end

@component function CalciumPool(; name, decay=100.0, Ca_init=0.0, topology=Scalar())
    @named port = CaPort(topology=topology)
    
    # If it's a function, we don't need the parameter, so we create a dummy.
    @parameters tau_Ca = (decay isa Function ? 0.0 : decay)
    
    if topology isa Scalar
        @variables Ca(t)=Ca_init
        vars = SymbolicT[Ca]
        init_conds = Dict(Ca => Ca_init)
    else
        @variables Ca(t)[1:topology.N] = fill(Ca_init, topology.N)
        vars = SymbolicT[Ca]
        init_conds = Dict(Ca => fill(Ca_init, topology.N))
    end

    # Dispatch the decay term based on type
    if decay isa Function
        decay_term = decay(Ca)
    else
        decay_term = .-Ca ./ tau_Ca
    end
    
    eqs = Equation[
        D(Ca) ~ decay_term .+ port.J_Ca,
        port.Ca ~ Ca
    ]
    
    # Only include the parameter if it was actually used
    params = decay isa Function ? SymbolicT[] : SymbolicT[tau_Ca]
    
    return System(eqs, t, vars, params; systems=[port], initial_conditions=init_conds, name=name)
end


@component function CaVChannel(; name, g, gates::Vector{<:GateSpec}, topology=Scalar(), 
                               conversion_factor=1.0, E_rev=nothing, Ca_out=3000.0, nernst_factor=13.0)
    if topology isa Scalar
        @named oneport = OnePort()
        @named ca_port = CaPort(topology=topology)
    else
        @named oneport = VectorizedOnePort(N=topology.N)
        @named ca_port = CaPort(topology=topology)
    end
    @unpack v, i = oneport
    
    @parameters g=g conversion_factor=conversion_factor
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{SymbolicT, Any}()
    
    params = SymbolicT[g, conversion_factor]
    
    # Handle E_rev (fixed vs dynamic Nernst)
    if isnothing(E_rev)
        @parameters Ca_out=Ca_out nernst_factor=nernst_factor
        # E_Ca = nernst_factor * ln(Ca_out / Ca_in)
        E_rev_expr = nernst_factor .* log.(Ca_out ./ ca_port.Ca)
        push!(params, Ca_out, nernst_factor)
    else
        @parameters E_rev=E_rev
        E_rev_expr = E_rev
        push!(params, E_rev)
    end
    
    conductance_factor = true
    for gate in gates
        if topology isa Scalar
            gate_var = only(@variables $(gate.name)(t))
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
            init_conds[gate_var] = gate.ic
        else
            gate_var = only(@variables $(gate.name)(t)[1:topology.N])
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:topology.N])
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:topology.N])
            init_conds[gate_var] = fill(gate.ic, topology.N)
        end
        
        push!(vars, gate_var, alpha_var, beta_var)
        alpha_expr, beta_expr = gate.dynamics(v)
        
        push!(eqs, alpha_var ~ alpha_expr)
        push!(eqs, beta_var ~ beta_expr)
        push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
        conductance_factor = conductance_factor .* (gate_var .^ gate.power)
    end
    
    # Electrical current uses the dynamic E_rev_expr
    push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev_expr))
    # Calcium flux (opposite sign to electrical current, scaled by factor)
    push!(eqs, ca_port.J_Ca ~ conversion_factor .* i)
    
    return extend(System(eqs, t, vars, params; 
                       systems=[ca_port], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end

@component function KCaChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, topology=Scalar())
    if topology isa Scalar
        @named oneport = OnePort()
        @named ca_port = CaPort(topology=topology)
    else
        @named oneport = VectorizedOnePort(N=topology.N)
        @named ca_port = CaPort(topology=topology)
    end
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{SymbolicT, Any}()
    
    # It senses calcium but doesn't contribute to the pool
    push!(eqs, ca_port.J_Ca ~ ground_current(topology))
    
    conductance_factor = true
    for gate in gates
        if topology isa Scalar
            gate_var = only(@variables $(gate.name)(t))
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
            init_conds[gate_var] = gate.ic
        else
            gate_var = only(@variables $(gate.name)(t)[1:topology.N])
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:topology.N])
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:topology.N])
            init_conds[gate_var] = fill(gate.ic, topology.N)
        end
        
        push!(vars, gate_var, alpha_var, beta_var)
        
        # Note: gate.dynamics now takes (v, Ca)
        alpha_expr, beta_expr = gate.dynamics(v, ca_port.Ca)
        
        push!(eqs, alpha_var ~ alpha_expr)
        push!(eqs, beta_var ~ beta_expr)
        push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
        conductance_factor = conductance_factor .* (gate_var .^ gate.power)
    end
    
    push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=[ca_port], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end
```

## File: src/components/channels.jl
```julia
# tempgates.jl

struct GateSpec{I<:Integer, T<:AbstractFloat, F<:Function}
    name::Symbol
    power::I
    ic::T
    dynamics::F 
end

@component function GenericChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, topology=Scalar())
    if topology isa Scalar
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=topology.N)
    end
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{Any, Any}()
    
    if isempty(gates)
        push!(eqs, i ~ g .* (v .- E_rev))
    else
        conductance_factor = true
        
        for gate in gates
            if topology isa Scalar
                gate_var = only(@variables $(gate.name)(t))
                alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
                beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
                init_conds[gate_var] = gate.ic
            else
                gate_var = only(@variables $(gate.name)(t)[1:topology.N])
                alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:topology.N])
                beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:topology.N])
                init_conds[gate_var] = fill(gate.ic, topology.N)
            end
            
            push!(vars, gate_var, alpha_var, beta_var)
            alpha_expr, beta_expr = gate.dynamics(v)
            
            push!(eqs, alpha_var ~ alpha_expr)
            push!(eqs, beta_var ~ beta_expr)
            push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
            
            conductance_factor = conductance_factor .* (gate_var .^ gate.power)
        end
        
        push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    end
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=System[], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end
```

## File: src/components/electrical.jl
```julia
# hi

@component function Ground(; name, topology=Scalar())
    if topology isa Scalar
        @named g = Pin()
        eqs = [g.v ~ 0]
    else
        @named g = VectorizedPin(N=topology.N)
        eqs = [g.v ~ zeros(Float64, topology.N)]
    end
    return System(eqs, t, SymbolicT[], SymbolicT[]; systems=[g], name=name)
end

@component function Capacitor(; name, C = 1.0, topology=Scalar())
    if topology isa Scalar
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=topology.N)
    end
    @unpack v, i = oneport
    @parameters C=C
    eqs = Equation[D(v) ~ i ./ C]
    return extend(System(eqs, t, SymbolicT[], [C]; systems=System[], name=name), oneport)
end

@component function CurrentSource(; name, topology=Scalar())
    if topology isa Scalar
        @named oneport = OnePort()
        @named I = RealInput()
    else
        @named oneport = VectorizedOnePort(N=topology.N)
        @named I = RealInputArray(nin=topology.N)
    end
    @unpack i = oneport
    
    eqs = Equation[i ~ -I.u]
    return extend(System(eqs, t, SymbolicT[], SymbolicT[]; systems=[I], name=name), oneport)
end

"""
fixed_reversal Component: A pure constant voltage source (Nernst battery).
"""
@component function FixedReversal(; name, E = 0.0)
    @named oneport = OnePort()
    @unpack v = oneport
    @parameters begin
        E = E
    end
    params = SymbolicT[]
    push!(params, E)
    vars = SymbolicT[]
    eqs = Equation[]
    push!(eqs, v ~ E)
    
    reversal_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        name
    )
    return extend(reversal_sys, oneport)
end

"""
SpikingCapacitor Component: Capacitor that automatically resets its voltage when a threshold is crossed 
"""
@component function SpikingCapacitor(; name, C = 10.0, V_th = -55.0, V_reset = -67.0, V_init = -65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters begin
        C = C
        V_th = V_th
        V_reset = V_reset
    end
    params = SymbolicT[]
    push!(params, C, V_th, V_reset)
    
    @variables begin
        v(t) = V_init
        V(t)
    end
    vars = SymbolicT[]
    push!(vars, v, V)

    eqs = Equation[
        D(v) ~ i / C,
        V ~ v
    ]
    
    root_eqs = Equation[v ~ V_th]
    affect = Equation[v ~ V_reset]
    events = [root_eqs => affect]
    
    lif_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        continuous_events = events,
        name
    )
    
    return extend(lif_sys, oneport)
end

@component function GapJunction(; name, R = 1.0)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport

    @parameters R = R
    params = SymbolicT[]
    push!(params, R)

    vars = SymbolicT[]

    eqs = Equation[]
    push!(eqs, i1 ~ (v1 - v2) / R)
    push!(eqs, i2 ~ -i1)

    return extend(System(eqs, t, vars, params; systems=System[], name=name), twoport)
end

@component function ChemicalSynapse(; name, g_max=2.0, τ=5.0, v_th=-20.0, w=0.5, E_rev=0.0)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport

    @parameters E_rev=E_rev g_max=g_max τ=τ v_th=v_th w=w
    params = SymbolicT[]
    push!(params, E_rev, g_max, τ, v_th, w)

    @variables s(t) = 0.0
    vars = SymbolicT[]
    push!(vars, s)

    eqs = Equation[]
    push!(eqs, i1 ~ 0.0)
    push!(eqs, D(s) ~ -s / τ)
    push!(eqs, i2 ~ (v2 - E_rev) * s * g_max)

    root_eqs = Equation[]
    push!(root_eqs, v1 ~ v_th)
    affect = Equation[]
    push!(affect, s ~ Pre(s) + w)
    events = [root_eqs => affect]

    return extend(System(eqs, t, vars, params; systems=System[], continuous_events=events, name=name), twoport)
end

# ==========================================
# VECTORIZED ELECTRICAL COMPONENTS
# ==========================================

@connector function VectorizedPin(; name, N::Int, v = nothing, i = nothing)
    vars = @variables begin
        v(t)[1:N] = v
        i(t)[1:N] = i, [connect = Flow]
    end
    return System(Equation[], t, vars, SymbolicT[]; name=name)
end

@component function VectorizedOnePort(; name, N::Int, v = nothing, i = nothing)
    pars = @parameters begin
    end
    systems = @named begin
        p = VectorizedPin(N=N)
        n = VectorizedPin(N=N)
    end
    vars = @variables begin
        v(t)[1:N] = v
        i(t)[1:N] = i
    end
    equations = Equation[
        v ~ p.v - n.v,
        collect(p.i .+ n.i .~ 0.0)...,  # splat the collected equations
        i ~ p.i,
    ]

    return System(equations, t, vars, pars; name, systems)
end
```

## File: src/components/synapses.jl
```julia
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

@component function ExpSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * (V_post - E_rev)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope]; 
                  systems=System[], name=name)
end

@component function AlphaSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s1(t)=0.0 s2(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s1) ~ -s1 / τ + σ(V_pre - V_th),
        D(s2) ~ -s2 / τ + s1,
        I_syn ~ g_max * s2 * (V_post - E_rev)
    ]
    return System(eqs, t, [s1, s2, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, slope]; systems=System[], name=name)
end

@component function NMDASynapse(; name, g_max=1.0, τ=100.0, E_rev=0.0, V_th=-20.0, 
                                  Mg_conc=1.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th Mg_conc=Mg_conc slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    mg_block(V) = 1.0 / (1.0 + Mg_conc * exp(-0.062 * V))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * mg_block(V_post) * (V_post - E_rev)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, Mg_conc, slope]; systems=System[], name=name)
end

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
        I_syn ~ g_max .* (V_post .- E_rev) .* synaptic_drive
    ]
    
    init_conds = Dict(s => zeros(N_pre))
    
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope, W];
                  systems=System[], 
                  initial_conditions=init_conds, 
                  name=name)
end
```

## File: src/library/HodgkinHuxley.jl
```julia
# ==========================================
# Standard Model Library
# ==========================================
module HodgkinHuxley
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, Scalar, Vectorized
    using ModelingToolkit: t_nounits as t, @named

    # Are these standard 1952 HH Gate Definitions? Forget where i found them. Check
    const na_m = v -> (
        0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),
        -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))
    )
    const na_h = v -> (
        0.25 .* exp.(-(v .+ 90.0) ./ 12.0),
        0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0)
    )
    const k_n = v -> (
        0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),
        -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))
    )

    const sodium_gates = [GateSpec(:m, 3, 0.0, na_m), GateSpec(:h, 1, 0.0, na_h)]
    const potassium_gates = [GateSpec(:n, 4, 0.0, k_n)]

    # Convenience constructors
    function SodiumChannel(; name, topology=Scalar(), g=120.0, E_rev=50.0)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=sodium_gates, topology=topology)
    end

    function PotassiumChannel(; name, topology=Scalar(), g=36.0, E_rev=-77.0)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=potassium_gates, topology=topology)
    end

    function LeakChannel(; name, topology=Scalar(), g=0.3, E_rev=-54.4)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=GateSpec[], topology=topology)
    end

    export SodiumChannel, PotassiumChannel, LeakChannel
end
```

## File: src/network.jl
```julia
struct Compartment
    sys::System
    interfaces::NamedTuple
    V_init::Float64
    topology::Union{Scalar, Vectorized}
end

struct Network
    sys::System
    inputs::Vector{Any}
end

struct SynapseSpec
    pre_V
    post_V
    post_I_syn
    synapse
    post_comp::Union{Compartment, Nothing} 
end

SynapseSpec(pre_V, post_V, post_I_syn, synapse) = SynapseSpec(pre_V, post_V, post_I_syn, synapse, nothing)

struct CouplingSpec
    comp_i::Compartment
    comp_j::Compartment
    coupling::System
end

# Ion config types
struct NoCalcium end
struct CalciumTracker
    decay::Union{Float64, Function}
    Ca_init::Float64
end

CalciumTracker(; decay=100.0, Ca_init=0.0) = CalciumTracker(decay, Ca_init)

# Ion dispatch
wire_ions!(eqs, systems, channels, ::NoCalcium, topology, name) = nothing
function wire_ions!(eqs, systems, channels, config::CalciumTracker, topology, name)
    # Pass decay to the CalciumPool
    ca_pool = CalciumPool(topology=topology, decay=config.decay, Ca_init=config.Ca_init, name=Symbol(name, :_ca_pool))
    push!(systems, ca_pool)
    
    ca_ports = System[ca_pool.port]
    for c in channels
        if hasproperty(c, :ca_port)
            push!(ca_ports, c.ca_port)
        end
    end
    push!(eqs, connect(ca_ports...))
end



# =========================================================
# 2. COMPARTMENT & CELL BUILDERS
# =========================================================

function build_compartment(capacitor, channels; name=:compartment, V_init=-65.0, 
                           topology=Scalar(), ion_config=NoCalcium())
    
    p, n = create_pins(topology)
    injector, syn_injector = create_injectors(topology)
    init_v = init_voltage(topology, V_init)
    
    vars = SymbolicT[]
    eqs  = Equation[]

    # 1. Connect all negative terminals together
    n_pins = Any[capacitor.n, injector.n, syn_injector.n, n]
    for c in channels
        push!(n_pins, c.n)
    end
    push!(eqs, connect(n_pins...))

    # 2. Connect all positive terminals together
    p_connections = System[capacitor, injector, syn_injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # 3. Expose boundary pin for acausal connections (gap junctions)
    push!(eqs, connect(p, capacitor.p))

    all_systems = System[capacitor, injector, syn_injector, p, n]
    append!(all_systems, channels)

    # 4. Wire ions (dispatches on config and topology)
    wire_ions!(eqs, all_systems, channels, ion_config, topology, name)

    sys = System(eqs, t, vars, SymbolicT[];
                 systems = all_systems,
                 initial_conditions = Dict(capacitor.v => init_v),
                 name)

    cap_name = nameof(capacitor)
    V_state  = getproperty(sys, cap_name).v

    interfaces = (
        V       = V_state,
        p_pin   = getproperty(sys, nameof(p)),
        n_pin   = getproperty(sys, nameof(n)),
        I_ext   = getproperty(sys, nameof(injector)).I.u,
        I_syn   = getproperty(sys, nameof(syn_injector)).I.u,
        cap_name = cap_name
    )
    return Compartment(sys, interfaces, V_init, topology)
end



# =========================================================
# 3. SYNAPSE WIRING
# =========================================================

"""
    wire_synapses!(eqs, systems, specs)

Wires a collection of SynapseSpecs into the network equations.
Pre-collects convergent synapses by target and writes one sum equation per target.
Returns the set of driven I_syn targets (for grounding the rest).
"""
function wire_synapses!(eqs::Vector{Equation}, systems::Vector{System},
                        specs::Vector{SynapseSpec})
    syn_by_target = Dict{SymbolicT, Vector{SymbolicT}}()
    driven_syn_targets = Set{SymbolicT}()
    block_driven_targets = Set{SymbolicT}()

    for spec in specs
        push!(systems, spec.synapse)
        
        if hasproperty(spec.synapse, :V_pre)
            push!(eqs, spec.synapse.V_pre ~ spec.pre_V)
        end
        if hasproperty(spec.synapse, :V_post)
            push!(eqs, spec.synapse.V_post ~ spec.post_V)
        end

        key = spec.post_I_syn
        haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
        push!(syn_by_target[key], spec.synapse.I_syn)
        push!(driven_syn_targets, key)
        
        if spec.post_I_syn isa AbstractArray
            push!(block_driven_targets, spec.post_I_syn)
        end
    end

    for (target, currents) in syn_by_target
        if length(currents) == 1
            push!(eqs, target ~ currents[1])
        else
            push!(eqs, target ~ reduce(+, currents))
        end
    end

    return driven_syn_targets, block_driven_targets
end


# =========================================================
# 4. NETWORK BUILDER
# =========================================================

function build_acausal_network(compartments::Vector{Compartment};
                                coupling_specs=CouplingSpec[],
                                synapse_specs=SynapseSpec[],
                                drivers=[],
                                name=:network)

    num_compartments = length(compartments)
    eqs = Equation[]
    all_systems = System[]

    for comp in compartments
        push!(all_systems, comp.sys)
    end

    driven_compartments = Set{Int}()
    gap_junctioned = Set{Int}()

    # 1. Ground each compartment individually (Dispatches on topology)
    for (i, comp) in enumerate(compartments)
        if haskey(comp.interfaces, :n_pin)
            gnd = create_ground(comp.topology, Symbol(:gnd_, i))
            push!(all_systems, gnd)
            push!(eqs, connect(gnd.g, comp.interfaces.n_pin))
        end
    end

    # 2. Driving stimuli
    for (target, stim) in drivers
        idx = target isa Compartment ? findfirst(==(target), compartments) : target
        push!(driven_compartments, idx)
        comp = compartments[idx]

        if haskey(comp.interfaces, :I_ext)
            if stim isa System
                push!(all_systems, stim)
                push!(eqs, comp.interfaces.I_ext ~ stim.output.u)
            elseif stim isa AbstractVector
                push!(eqs, comp.interfaces.I_ext ~ stim)
            elseif stim isa Number
                push!(eqs, comp.interfaces.I_ext ~ broadcast_stim(comp.topology, stim))
            end
        end
    end

    # 3. Ground undriven I_ext
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_ext) && !(i in driven_compartments)
            push!(eqs, comp.interfaces.I_ext ~ ground_current(comp.topology))
        end
    end

    # 4. Wire gap junctions via p_pin
    for (i, spec) in enumerate(coupling_specs)
        push!(all_systems, spec.coupling)
        
        if haskey(spec.comp_i.interfaces, :p_pin) && hasproperty(spec.coupling, :p1)
            push!(eqs, connect(spec.comp_i.interfaces.p_pin, spec.coupling.p1))
            push!(eqs, connect(spec.coupling.n1, spec.comp_i.interfaces.n_pin))
        end
        
        if haskey(spec.comp_j.interfaces, :p_pin) && hasproperty(spec.coupling, :p2)
            push!(eqs, connect(spec.comp_j.interfaces.p_pin, spec.coupling.p2))
            push!(eqs, connect(spec.coupling.n2, spec.comp_j.interfaces.n_pin))
        end
        
        push!(gap_junctioned, findfirst(==(spec.comp_i), compartments))
        push!(gap_junctioned, findfirst(==(spec.comp_j), compartments))
    end

    # 5. Identify block-synapsed compartments by index
    block_synapsed_compartments = Set{Int}()
    for spec in synapse_specs
        if spec.post_I_syn isa AbstractArray && spec.post_comp !== nothing
            idx = findfirst(==(spec.post_comp), compartments)
            if idx !== nothing
                push!(block_synapsed_compartments, idx)
            end
        end
    end

    # 6. Wire synapses
    driven_syn_targets, block_driven_targets = wire_synapses!(eqs, all_systems, synapse_specs)

    # 7. Ground non-synapsed I_syn (Dispatches on topology)
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_syn)
            if comp.interfaces.I_syn in block_driven_targets
                continue
            end
            ground_undriven_syn!(eqs, comp.topology, comp.interfaces.I_syn, driven_syn_targets)
        end
    end

    # 8. Ground non-gap-junctioned p_pin.i
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :p_pin) && !(i in gap_junctioned)
            push!(eqs, comp.interfaces.p_pin.i ~ ground_current(comp.topology))
        end
    end

    net_sys = System(eqs, t, SymbolicT[], SymbolicT[];
                     systems = all_systems, name = name)
                     
    return Network(net_sys, SymbolicT[])
end


function build_synapse_block(pre_comp, post_comp, W; name, 
                             synapse_type=VectorizedExpSynapse, kwargs...)
    N_pre  = size(W, 2)
    N_post = size(W, 1)
    syn = synapse_type(N_pre=N_pre, N_post=N_post, W=W; name=name, kwargs...)
    return SynapseSpec(pre_comp.interfaces.V, post_comp.interfaces.V,
                       post_comp.interfaces.I_syn, syn, post_comp)
end
```

## File: src/topology.jl
```julia
struct Scalar end
struct Vectorized
    N::Int
end

# Topology helper functions
get_N(::Scalar) = nothing
get_N(v::Vectorized) = v.N

init_voltage(::Scalar, V_init) = V_init
init_voltage(v::Vectorized, V_init) = fill(V_init, v.N)

function create_pins(::Scalar)
    @named p = Pin(); @named n = Pin()
    return (p, n)
end
function create_pins(v::Vectorized)
    @named p = VectorizedPin(N=v.N); @named n = VectorizedPin(N=v.N)
    return (p, n)
end

function create_injectors(::Scalar)
    @named injector = CurrentSource(); @named syn_injector = CurrentSource()
    return (injector, syn_injector)
end
function create_injectors(v::Vectorized)
    @named injector = CurrentSource(topology=v)
    @named syn_injector = CurrentSource(topology=v)
    return (injector, syn_injector)
end

# Network grounding helpers
create_ground(::Scalar, name) = Ground(name=name)
create_ground(v::Vectorized, name) = Ground(topology=v, name=name)

ground_current(::Scalar) = 0.0
ground_current(v::Vectorized) = zeros(Float64, v.N)

broadcast_stim(::Scalar, stim) = stim
broadcast_stim(v::Vectorized, stim) = fill(stim, v.N)

# Synapse grounding helpers
function ground_undriven_syn!(eqs, ::Scalar, I_syn, driven_syn_targets)
    if !(I_syn in driven_syn_targets)
        push!(eqs, I_syn ~ 0.0)
    end
end
function ground_undriven_syn!(eqs, v::Vectorized, I_syn, driven_syn_targets)
    for j in 1:v.N
        i_syn_j = I_syn[j]
        if !(i_syn_j in driven_syn_targets)
            push!(eqs, i_syn_j ~ 0.0)
        end
    end
end
```

## File: src/connections.jl
```julia
# connections.jl (Replace the helper functions at the top with these)


# =========================================================
# 1. STRUCT DEFINITIONS & TOPOLOGY HELPERS
# =========================================================
```

## File: src/MTKNeuralToolkit.jl
```julia
module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput, Constant, RealOutput, RealInputArray
import ModelingToolkitStandardLibrary.Electrical: OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, SymbolicT, ImperativeAffect
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
import SymbolicUtils: scalarize
import Symbolics: Sym, Num

# ==========================================
# 1. Core Framework
# ==========================================
include("topology.jl")
export Scalar, Vectorized

include("components/electrical.jl")

include("components/channels.jl")
export Ground, Capacitor, CurrentSource, GenericChannel, GateSpec

include("components/calcium.jl")
include("components/synapses.jl")
include("network.jl")

export build_compartment, build_acausal_network, build_synapse_block

export Compartment, Network, SynapseSpec, CouplingSpec
export CaVChannel, KCaChannel, CalciumPool, CalciumTracker, NoCalcium, CaPort
export ExpSynapse, VectorizedExpSynapse

export ContinuousLIFChannel

# ==========================================
# 2. Standard Model Library (Submodules)
# ==========================================
include("library/HodgkinHuxley.jl")
export HodgkinHuxley

include("library/ContinuousSpikers.jl")
export ContinuousSpikers


end
```

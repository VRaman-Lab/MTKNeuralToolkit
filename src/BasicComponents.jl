"""
Soma Component: Represents a pure physical lipid bilayer membrane patch.
"""
@component function Capacitor(; name, C = 1.0, V_init = -65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    @parameters begin
        C = C
    end
    params = SymbolicT[]
    push!(params, C)
    
    @variables begin
        V(t) = V_init
    end
    vars = SymbolicT[]
    push!(vars, V)
    
    eqs = Equation[]
    push!(eqs, D(v) ~ i / C)
    push!(eqs, V ~ v)
    
    cap_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        name
    )
    return extend(cap_sys, oneport)
end

"""
CurrentSource Component: Converts a causal RealInput signal (u) 
into an acausal electrical current (i) injecting into a physical Node.
"""
@component function CurrentSource(; name)
    @named oneport = OnePort()
    @unpack i = oneport
    @named I = RealInput()
    
    vars = SymbolicT[]
    params = SymbolicT[]
    eqs = Equation[]
    push!(eqs, i ~ -I.u)
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    # We cast 'I' into a Vector{System} instead of leaving it as an untyped literal array
    subsystems = System[]
    push!(subsystems, I)
    
    source_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = subsystems, 
        initial_conditions, 
        guesses, 
        name
    )
    return extend(source_sys, oneport)
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
LIFCapacitor Component: Capacitor that automatically resets its voltage when a threshold is crossed 
"""
@component function LIFCapacitor(; name, C = 10.0, V_th = -55.0, V_reset = -67.0, V_init = -65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters begin
        C = C
        V_th = V_th
        V_reset = V_reset
    end
    params = SymbolicT[C, V_th, V_reset]
    
    @variables begin
        # Bind the incoming V_init default directly to the true differential state
        v(t) = V_init
        V(t)
    end
    # Include both v and V in the structural variables array
    vars = SymbolicT[v, V]
    
    eqs = Equation[
        D(v) ~ i / C,
        V ~ v
    ]
    
    root_eqs = Equation[v ~ V_th]
    affect = Equation[v ~ V_reset]
    events = root_eqs => affect
    
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


@component function AlphaSynapse(; name, g_max=3.0, τ=5.0, E_rev=0.0, v_th=-20.0, w=1.0)
    # Only s(t) gets a constant default because it's a differential state.
    # V_pre, V_post, and I_syn are algebraic/boundary variables determined by connections.
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev v_th=v_th w=w

    eqs = [
        D(s) ~ -s / τ,
        I_syn ~ (V_post - E_rev) * s * g_max
    ]
    
    continuous_events = [[V_pre ~ v_th] => [s ~ Pre(s) + w]]
    
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, v_th, w]; continuous_events, name)
end

export AlphaSynapse

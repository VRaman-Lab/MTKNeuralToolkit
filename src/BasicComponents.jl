"""
Soma Component: Represents a pure physical lipid bilayer membrane patch.
"""
@component function Capacitor(; name, C = 1.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    @parameters begin
        C = C
    end
    params = SymbolicT[]
    push!(params, C)
    
    vars = SymbolicT[]
    
    eqs = Equation[]
    push!(eqs, D(v) ~ i / C)
    
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
    params = params = SymbolicT[]
    push!(params, C, V_th, V_reset)
    
    @variables begin
        # Bind the incoming V_init default directly to the true differential state
        v(t) = V_init
        V(t)
    end
    # Include both v and V in the structural variables array
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
    # The current flowing into port 1 is driven by the voltage difference.
    # By conservation of current, what goes into port 1 must come out of port 2.
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

    # Events should also be built cleanly
    root_eqs = Equation[]
    push!(root_eqs, v1 ~ v_th)
    affect = Equation[]
    push!(affect, s ~ Pre(s) + w)
    events = [root_eqs => affect]

    return extend(System(eqs, t, vars, params; systems=System[], continuous_events=events, name=name), twoport)
end

@component function AlphaSynapse(; name, g_max=3.0, τ=5.0, E_rev=0.0, v_th=-20.0, w=1.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev v_th=v_th w=w

    vars = SymbolicT[]
    push!(vars, s, I_syn, V_pre, V_post)

    params = SymbolicT[]
    push!(params, g_max, τ, E_rev, v_th, w)

    eqs = Equation[]
    push!(eqs, D(s) ~ -s / τ)
    push!(eqs, I_syn ~ (V_post - E_rev) * s * g_max)

    # Build event equations as explicitly typed Equation[] vectors
    root_eqs = Equation[]
    push!(root_eqs, V_pre ~ v_th)

    affect = Equation[]
    push!(affect, s ~ Pre(s) + w)
    push!(affect, V_pre ~ Pre(V_pre))   # Lock pre-synaptic voltage
    push!(affect, V_post ~ Pre(V_post)) # Lock post-synaptic voltage

    events = Any[] 
    push!(events, root_eqs => affect)

    # Explicitly pass systems=System[]
    return System(eqs, t, vars, params; systems=System[], continuous_events=events, name=name)
end


function spike_affect!(mod, obs, ctx, integ)
    j = ctx.j
    W = ctx.W
    N = ctx.N

    S_new = copy(mod.S)
    for i in 1:N
        S_new[j, i] += W[j, i]
    end
    return (; S = S_new)
end

@component function VectorizedAlphaSynapse(; name, N::Int, W::Matrix{Float64}, tau::Matrix{Float64}, g_max::Matrix{Float64},
E_rev=0.0, v_th=-20.0)
    # I_inj has NO default value, because it is an algebraic variable
    @variables V_vec(t)[1:N] I_inj(t)[1:N] S(t)[1:N, 1:N]=zeros(Float64, N, N)
    @parameters tau_p[1:N, 1:N]=tau g_max_p[1:N, 1:N]=g_max E_rev_p=E_rev v_th_p=v_th

    eqs = Equation[]
    push!(eqs, D(S) ~ -S ./ tau_p)

    for i in 1:N
        rhs = Num(0.0)
        for j in 1:N
            rhs += (V_vec[i] - E_rev_p) * S[j, i] * g_max_p[j, i]
        end
        push!(eqs, I_inj[i] ~ rhs)
    end

    events = Any[]
    for j in 1:N
        event = [V_vec[j] ~ v_th_p] => ImperativeAffect(
            spike_affect!,
            modified = (; S),
            observed = (;),
            ctx = (j=j, W=W, N=N)
        )
        push!(events, event)
    end

    # Type-stable SymbolicT[] vectors for fast precompilation
    vars = SymbolicT[]
    push!(vars, S)
    push!(vars, I_inj)
    push!(vars, V_vec)

    params = SymbolicT[]
    push!(params, tau_p)
    push!(params, g_max_p)
    push!(params, E_rev_p)
    push!(params, v_th_p)

    # No guesses needed, the system is perfectly balanced
    return System(eqs, t, vars, params;
                  continuous_events=events,
                  systems = System[],
                  name)
end

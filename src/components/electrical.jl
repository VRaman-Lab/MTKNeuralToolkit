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

@component function Capacitor(; name, C = 1.0, topology=Scalar(), geometry=NoGeometry())
    C_val = get_capacitance(C, geometry) # Dispatch handles the math
    
    if topology isa Scalar
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=topology.N)
    end
    @unpack v, i = oneport
    @parameters C=C_val
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

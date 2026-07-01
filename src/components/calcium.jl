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

    if decay isa Function
        decay_term = decay(Ca)
    else
        decay_term = .-Ca ./ tau_Ca
    end
    
    eqs = Equation[
        D(Ca) ~ decay_term .+ port.J_Ca,
        port.Ca ~ Ca
    ]
    
    params = decay isa Function ? SymbolicT[] : SymbolicT[tau_Ca]
    
    return System(eqs, t, vars, params; systems=[port], initial_conditions=init_conds, name=name)
end


@component function CaVChannel(; name, g, gates::Vector{<:GateSpec}, topology=Scalar(), 
                               conversion_factor=1.0, E_rev=nothing, Ca_out=3000.0, nernst_factor=13.0,
                               geometry=NoGeometry(), tauCa=nothing)
    # 1. Scale geometry
    g_val = get_conductance(g, geometry)
    conv_val = get_ca_conversion_factor(conversion_factor, geometry, tauCa)
    
    if topology isa Scalar
        @named oneport = OnePort()
        @named ca_port = CaPort(topology=topology)
    else
        @named oneport = VectorizedOnePort(N=topology.N)
        @named ca_port = CaPort(topology=topology)
    end
    @unpack v, i = oneport
    
    @parameters g=g_val conversion_factor=conv_val
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{SymbolicT, Any}()
    
    params = SymbolicT[g, conversion_factor]
    
    if isnothing(E_rev)
        @parameters Ca_out=Ca_out nernst_factor=nernst_factor
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
    
    # Ensure we use the symbolic `g` and `conversion_factor`
    push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev_expr))
    push!(eqs, ca_port.J_Ca ~ conversion_factor .* i)
    
    return extend(System(eqs, t, vars, params; 
                       systems=[ca_port], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end

@component function KCaChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, topology=Scalar(), geometry=NoGeometry())
    # 1. Scale geometry
    g_val = get_conductance(g, geometry)
    
    if topology isa Scalar
        @named oneport = OnePort()
        @named ca_port = CaPort(topology=topology)
    else
        @named oneport = VectorizedOnePort(N=topology.N)
        @named ca_port = CaPort(topology=topology)
    end
    @unpack v, i = oneport
    
    @parameters g=g_val E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{SymbolicT, Any}()
    
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
        
        alpha_expr, beta_expr = gate.dynamics(v, ca_port.Ca)
        
        push!(eqs, alpha_var ~ alpha_expr)
        push!(eqs, beta_var ~ beta_expr)
        push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
        conductance_factor = conductance_factor .* (gate_var .^ gate.power)
    end
    
    # Ensure we use the symbolic `g`
    push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=[ca_port], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end

@component function CaVChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, N::Union{Int, Nothing}=nothing, conversion_factor=1.0)
    if isnothing(N)
        @named oneport = OnePort()
        @named ca_port = CaPort()
    else
        @named oneport = VectorizedOnePort(N=N)
        @named ca_port = CaPort(N=N)
    end
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev conversion_factor=conversion_factor
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{Any, Any}()
    
    conductance_factor = true
    for gate in gates
        if isnothing(N)
            gate_var = only(@variables $(gate.name)(t))
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
            init_conds[gate_var] = gate.ic
        else
            gate_var = only(@variables $(gate.name)(t)[1:N])
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:N])
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:N])
            init_conds[gate_var] = fill(gate.ic, N)
        end
        
        push!(vars, gate_var, alpha_var, beta_var)
        alpha_expr, beta_expr = gate.dynamics(v)
        
        push!(eqs, alpha_var ~ alpha_expr)
        push!(eqs, beta_var ~ beta_expr)
        push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
        conductance_factor = conductance_factor .* (gate_var .^ gate.power)
    end
    
    # Electrical current
    push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    # Calcium flux (opposite sign to electrical current, scaled by factor)
    push!(eqs, ca_port.J_Ca ~ .-conversion_factor .* i)
    
    return extend(System(eqs, t, vars, [g, E_rev, conversion_factor]; 
                       systems=[ca_port], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end


@component function KCaChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named oneport = OnePort()
        @named ca_port = CaPort()
    else
        @named oneport = VectorizedOnePort(N=N)
        @named ca_port = CaPort(N=N)
    end
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{Any, Any}()
    
    # It senses calcium but doesn't contribute to the pool
    push!(eqs, ca_port.J_Ca ~ 0.0)
    
    conductance_factor = true
    for gate in gates
        if isnothing(N)
            gate_var = only(@variables $(gate.name)(t))
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
            init_conds[gate_var] = gate.ic
        else
            gate_var = only(@variables $(gate.name)(t)[1:N])
            alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:N])
            beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:N])
            init_conds[gate_var] = fill(gate.ic, N)
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



export CaVChannel, KCaChannel

struct GateSpec{I<:Integer, T<:AbstractFloat, F<:Function}
    name::Symbol
    power::I
    ic::T
    # A function taking voltage `v` and returning a tuple: (alpha_expr, beta_expr)
    dynamics::F 
end

@component function GenericChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=N)
    end
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{Any, Any}()
    
    if isempty(gates)
        # Pure leak channel (avoids broadcasting edge cases with empty gates)
        push!(eqs, i ~ g .* (v .- E_rev))
    else
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
        
        push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    end
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=System[], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end

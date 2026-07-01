struct GateSpec{I<:Integer, T<:AbstractFloat, F<:Function}
    name::Symbol
    power::I
    ic::T
    dynamics::F 
end

# Convert (inf, tau) -> (alpha, beta) where alpha = inf/tau and beta = (1-inf)/tau

InfTau(inf_fn, tau_fn) = v -> (inf_fn(v) ./ tau_fn(v), (1.0 .- inf_fn(v)) ./ tau_fn(v))
InfTauCa(inf_fn, tau_fn) = (v, ca) -> (inf_fn(v, ca) ./ tau_fn(v), (1.0 .- inf_fn(v, ca)) ./ tau_fn(v))

@component function GenericChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, topology=Scalar(), geometry=NoGeometry())
    g_val = get_conductance(g, geometry) # Dispatch handles the math
    
    if topology isa Scalar
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=topology.N)
    end
    @unpack v, i = oneport
    
    @parameters g=g_val E_rev=E_rev
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


@component function ContinuousLIFChannel(; name, g_L=0.1, E_L=-70.0, V_th=-50.0, Δ_T=2.0, topology=Scalar())
    if topology isa Scalar
        @named oneport = OnePort()
        @unpack v, i = oneport
        
        @parameters g_L=g_L E_L=E_L V_th=V_th Δ_T=Δ_T
        params = SymbolicT[g_L, E_L, V_th, Δ_T]
        
        vars = SymbolicT[]
        
        # Standard scalar math
        reset_current = g_L * Δ_T * exp((v - V_th) / Δ_T)
        eqs = Equation[
            i ~ g_L * (v - E_L) + reset_current
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
    else
        N = topology.N
        @named oneport = VectorizedOnePort(N=N)
        @unpack v, i = oneport
        
        @parameters g_L=g_L E_L=E_L V_th=V_th Δ_T=Δ_T
        params = SymbolicT[g_L, E_L, V_th, Δ_T]
        
        vars = SymbolicT[]
        
        # Use scalar * array (g_L * ...) instead of broadcast (g_L .* ...) 
        # to avoid Symbolics BroadcastBuffer errors.
        diff = v .- V_th
        leak_current = g_L * (v .- E_L)
        reset_current = (g_L * Δ_T) * exp.(diff ./ Δ_T)
        
        eqs = Equation[
            i ~ leak_current .+ reset_current
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
    end
end



"""
NOT GOING WITH THIS. would have to maintain vectorised callbacks which are a PITA.
keeping as a reference.
"""
function VectorSynapsePopulation(; name, N_conns, g_max_vec, τ_vec, v_th_vec, w_vec)
    @parameters begin
        g_max[1:N_conns] = g_max_vec
        τ[1:N_conns] = τ_vec
        v_th[1:N_conns] = v_th_vec
        w[1:N_conns] = w_vec
    end
    params = SymbolicT[]
    for idx in 1:N_conns
        push!(params, g_max[idx])
        push!(params, τ[idx])
        push!(params, v_th[idx])
        push!(params, w[idx])
    end
    
    @variables begin
        s(t)[1:N_conns]
        V_pre(t)[1:N_conns]
        V_post(t)[1:N_conns]
        I_syn(t)[1:N_conns]
    end
    vars = SymbolicT[]
    for idx in 1:N_conns
        push!(vars, s[idx])
        push!(vars, V_pre[idx])
        push!(vars, V_post[idx])
        push!(vars, I_syn[idx])
    end
    
    # --- FIXED INITIAL CONDITIONS FOR ATOMIC ARRAYS ---
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    # We assign a numeric vector to the parent atomic array symbol 's' directly
    initial_conditions[s] = zeros(N_conns) 
    # ---------------------------------------------------
    
    eqs = Equation[]
    for idx in 1:N_conns
        push!(eqs, D(s[idx]) ~ -s[idx] / τ[idx])
        push!(eqs, I_syn[idx] ~ V_post[idx] * s[idx] * g_max[idx]) 
    end
    
    root_eqs = Equation[]
    affect = Equation[]
    for idx in 1:N_conns
        push!(root_eqs, V_pre[idx] ~ v_th[idx])
        push!(affect, s[idx] ~ Pre(s[idx]) + w[idx])
    end
    
    events = root_eqs => affect
    
    return System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        initial_conditions, 
        guesses, 
        continuous_events = events,
        name
    )
end

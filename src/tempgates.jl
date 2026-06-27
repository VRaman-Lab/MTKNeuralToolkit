using Symbolics: variable
struct GateSpec
    name::Symbol
    power::Int
    ic::Float64
    # A function taking voltage `v` and returning a tuple: (alpha_expr, beta_expr)
    dynamics::Function 
end

@component function GenericChannel(; name, g, E_rev, gates::Vector{GateSpec})
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    
    # Dictionary to cleanly hold initial conditions for dynamically created vars
    init_conds = Dict{Any, Any}()
    
    conductance_factor = Num(1.0)
    
    for gate in gates
        # Dynamically create the gate variable and its rate variables
        gate_var = only(@variables $(gate.name)(t))
        alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
        beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
        
        push!(vars, gate_var, alpha_var, beta_var)
        init_conds[gate_var] = gate.ic
        
        # Call the user's function to get the symbolic alpha/beta equations
        alpha_expr, beta_expr = gate.dynamics(v)
        
        push!(eqs, alpha_var ~ alpha_expr)
        push!(eqs, beta_var ~ beta_expr)
        push!(eqs, D(gate_var) ~ alpha_var * (1 - gate_var) - beta_var * gate_var)
        
        # Multiply into the overall conductance (e.g., m^3 * h^1)
        conductance_factor *= gate_var ^ gate.power
    end
    
    # Final Ohm's law using driving force
    push!(eqs, i ~ g * conductance_factor * (v - E_rev))
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=System[], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end


@component function InlinedHHNeuron(; name, C=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.4, V_init=-65.0)
    @named oneport = OnePort()
    @unpack v, i, p, n = oneport
    @named injector = CurrentSource()
    @named ground = Ground()

    @parameters C=C g_Na=g_Na g_K=g_K g_L=g_L E_Na=E_Na E_K=E_K E_L=E_L
    params = SymbolicT[]
    push!(params, C, g_Na, g_K, g_L, E_Na, E_K, E_L)

    @variables begin
        V(t) = V_init
        m(t) = 0.0
        h(t) = 1.0
        n_gate(t) = 0.0
        I_Na(t)
        I_K(t)
        I_L(t)
        αₘ(t), βₘ(t)
        αₕ(t), βₕ(t)
        αₙ(t), βₙ(t)
    end
    vars = SymbolicT[]
    push!(vars, V, m, h, n_gate, I_Na, I_K, I_L, αₘ, βₘ, αₕ, βₕ, αₙ, βₙ)
    eqs = Equation[]
    push!(eqs, V ~ v)

    # Ground the membrane and the injector pins to prevent floating singularities
    push!(eqs, connect(ground.g, n))
    push!(eqs, connect(ground.g, injector.n))
    push!(eqs, connect(ground.g, injector.p))
    push!(eqs, i ~ p.i)

    # Na gating
    push!(eqs, αₘ ~ 0.182 * ((v - E_Na) + 35.0) / (1.0 - exp(-((v - E_Na) + 35.0) / 9.0)))
    push!(eqs, βₘ ~ -0.124 * ((v - E_Na) + 35.0) / (1.0 - exp(((v - E_Na) + 35.0) / 9.0)))
    push!(eqs, αₕ ~ 0.25 * exp(-((v - E_Na) + 90.0) / 12.0))
    push!(eqs, βₕ ~ 0.25 * (exp(((v - E_Na) + 62.0) / 6.0)) / exp(-((v - E_Na) + 90.0) / 12.0))
    push!(eqs, D(m) ~ αₘ * (1 - m) - βₘ * m)
    push!(eqs, D(h) ~ αₕ * (1 - h) - βₕ * h)
    push!(eqs, I_Na ~ g_Na * m^3 * h * (v - E_Na))

    # K gating
    push!(eqs, αₙ ~ 0.02 * ((v - E_K) - 25.0) / (1.0 - exp(-((v - E_K) - 25.0) / 9.0)))
    push!(eqs, βₙ ~ -0.002 * ((v - E_K) - 25.0) / (1.0 - exp(((v - E_K) - 25.0) / 9.0)))
    push!(eqs, D(n_gate) ~ αₙ * (1 - n_gate) - βₙ * n_gate)
    push!(eqs, I_K ~ g_K * n_gate^4 * (v - E_K))

    # Leak
    push!(eqs, I_L ~ g_L * (v - E_L))

    # Membrane equation: 'i' is acausal current, 'injector.I.u' is causal stimulus
    push!(eqs, C * D(v) ~ i + injector.I.u - I_Na - I_K - I_L)

    return extend(System(eqs, t, vars, params; systems=[injector, ground], name=name), oneport)
end

@component function VectorizedHHNeuron(; name, N::Int, C=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0,
E_L=-54.4, V_init=-65.0)
    # Parameters (scalars are automatically broadcasted by MTK if applied to arrays)
    @parameters C=C g_Na=g_Na g_K=g_K g_L=g_L E_Na=E_Na E_K=E_K E_L=E_L V_init=V_init
    params = SymbolicT[]
    push!(params, C, g_Na, g_K, g_L, E_Na, E_K, E_L, V_init)

    # Array Variables
    @variables begin
        V(t)[1:N] = fill(V_init, N)
        I_inj(t)[1:N] 
        m(t)[1:N] = zeros(Float64, N)
        h(t)[1:N] = ones(Float64, N)
        n_gate(t)[1:N] = zeros(Float64, N)
        I_Na(t)[1:N]
        I_K(t)[1:N]
        I_L(t)[1:N]
        αₘ(t)[1:N]
        βₘ(t)[1:N]
        αₕ(t)[1:N]
        βₕ(t)[1:N]
        αₙ(t)[1:N]
        βₙ(t)[1:N]
    end

    vars = SymbolicT[]
    push!(vars, V, I_inj, m, h, n_gate, I_Na, I_K, I_L, αₘ, βₘ, αₕ, βₕ, αₙ, βₙ)

    eqs = Equation[]

    # Na gating (using broadcasting .*)
    push!(eqs, αₘ ~ 0.182 .* (V .- E_Na .+ 35.0) ./ (1.0 .- exp.(-(V .- E_Na .+ 35.0) ./ 9.0)))
    push!(eqs, βₘ ~ -0.124 .* (V .- E_Na .+ 35.0) ./ (1.0 .- exp.((V .- E_Na .+ 35.0) ./ 9.0)))
    push!(eqs, αₕ ~ 0.25 .* exp.(-(V .- E_Na .+ 90.0) ./ 12.0))
    push!(eqs, βₕ ~ 0.25 .* (exp.((V .- E_Na .+ 62.0) ./ 6.0)) ./ exp.(-(V .- E_Na .+ 90.0) ./ 12.0))
    push!(eqs, D(m) ~ αₘ .* (1.0 .- m) .- βₘ .* m)
    push!(eqs, D(h) ~ αₕ .* (1.0 .- h) .- βₕ .* h)
    push!(eqs, I_Na ~ g_Na .* (m .^ 3) .* h .* (V .- E_Na))

    # K gating
    push!(eqs, αₙ ~ 0.02 .* (V .- E_K .- 25.0) ./ (1.0 .- exp.(-(V .- E_K .- 25.0) ./ 9.0)))
    push!(eqs, βₙ ~ -0.002 .* (V .- E_K .- 25.0) ./ (1.0 .- exp.((V .- E_K .- 25.0) ./ 9.0)))
    push!(eqs, D(n_gate) ~ αₙ .* (1.0 .- n_gate) .- βₙ .* n_gate)
    push!(eqs, I_K ~ g_K .* (n_gate .^ 4) .* (V .- E_K))

    # Leak
    push!(eqs, I_L ~ g_L .* (V .- E_L))

    # Membrane equation
    push!(eqs, D(V) ~ (I_inj .- I_Na .- I_K .- I_L) ./ C)

    return System(eqs, t, vars, params; systems=System[], name=name)
end


@component function STDPSynapse(; name, N::Int, W_init::Matrix{Float64}, A_plus=0.01, A_minus=0.01, tau_plus=20.0,
tau_minus=20.0, v_th=-20.0)
    @variables V_vec(t)[1:N] I_inj(t)[1:N] W(t)[1:N, 1:N]=W_init t_pre(t)[1:N]=fill(-1000.0, N) t_post(t)[1:N]=fill(-1000.0,
N)
    @parameters A_plus=A_plus A_minus=A_minus tau_plus=tau_plus tau_minus=tau_minus v_th_p=v_th

    # Synaptic conductance based on dynamic weight W
    eqs = Equation[]
    push!(eqs, I_inj ~ V_vec .* W)  # Simplified for example

    events = Any[]
    for j in 1:N # Pre-synaptic spikes
        root_eqs = [V_vec[j] ~ v_th_p]
        affect = [
            t_pre[j] ~ t,
            W[j, :] ~ clamp.(Pre(W[j, :]) .+ A_plus .* exp.(-(t .- Pre(t_post[:])) ./ tau_plus), 0.0, 1.0)
        ]
        push!(events, root_eqs => affect)
    end

    for i in 1:N # Post-synaptic spikes
        root_eqs = [V_vec[i] ~ v_th_p]
        affect = [
            t_post[i] ~ t,
            W[:, i] ~ clamp.(Pre(W[:, i]) .- A_minus .* exp.(-(t .- Pre(t_pre[:])) ./ tau_minus), 0.0, 1.0)
        ]
        push!(events, root_eqs => affect)
    end

    return System(eqs, t, [V_vec, I_inj, W, t_pre, t_post], [A_plus, A_minus, tau_plus, tau_minus, v_th_p]; continuous_events=events, name=name)
end

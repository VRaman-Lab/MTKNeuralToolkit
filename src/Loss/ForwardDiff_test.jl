function Forwardiff_test(system, ref_sol, prob)
    ground_sol = generate_groundtruth_system(ref_sol)
    tsteps = ground_sol.t
    
    param_syms = parameters(prob.f.sys)
    p_array = [Float64(prob.ps[sym]) for sym in param_syms]
    
    g_max_sym = system.s12_LIF1.g_max
    g_max_idx = findfirst(s -> isequal(s, g_max_sym), param_syms)
    
    f_no_cb = ODEFunction(prob.f.f)

    prob_plain = ODEProblem(f_no_cb, prob.u0, prob.tspan, p_array)
    
    state_sym = system.IF2.IF2.oneport.v
    state_idx = variable_index(prob, state_sym)

    p0 = [p_array[g_max_idx]]
    print(p0)

    optfn = OptimizationFunction(loss_forwards, Optimization.AutoForwardDiff(; chunksize = 1))
    optprob = OptimizationProblem(
        optfn, p0, (prob_plain, tsteps, ground_sol, g_max_idx, state_idx),
        lb = 0.0, ub = 100.0
    )
    
    sol = solve(optprob, GradientDescent(); time_limit=10)
    print(sol.u, sol.minimum)
    return sol.u[1], sol.minimum
end

function loss_forwards(x, p)
    prob, tsteps, truth, g_max_idx, state_idx = p

    T = eltype(x)
    p_new = Vector{T}(undef, length(prob.p))
    for i in 1:length(prob.p)
        p_new[i] = prob.p[i]
    end

    p_new[g_max_idx] = x[1]

    #println(p_new[g_max_idx])

    newprob = ODEProblem(prob.f, prob.u0, prob.tspan, p_new)

    sol = solve(newprob, Tsit5(); saveat = tsteps, save_end = false)

    pred = sol(tsteps)[state_idx, :]
    truth_vals = truth[state_idx, :]
    
    loss_val = mean((truth_vals .- pred).^2)
    
    return loss_val
end

function generate_groundtruth_system(ref_sol)
    @named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20), 20.0, 0.0))
    neurons = [
        MTKNeuralToolkit.build_LIF(inp; name=:IF1),
        MTKNeuralToolkit.build_LIF(; name=:IF2) 
    ]
    connections = Dict(
        (1, 2) => [(type=:LIF, weight=0.0)]
    )
    ground_sys = build_network(connections, neurons)
    ground_prob = ODEProblem(ground_sys, Pair[], (0.0, 200.0))
    ground_sol = solve(ground_prob, Tsit5(); saveat=ref_sol.t)
    return ground_sol
end

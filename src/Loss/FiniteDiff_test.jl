
"
This is all test code and is in the works (Ella dissertation)
"

function optim_test(system, ref_sol, prob)

    loss_arr = []
    values_arr = [] 

    ground_sol = generate_groundtruth_system(ref_sol)
    tsteps = ground_sol.t
    
    p0 = get_weights(system, prob)

    optfn = OptimizationFunction(loss_finitediff, Optimization.AutoFiniteDiff())
    
    optprob = OptimizationProblem(
    optfn, p0, (prob, tsteps, ground_sol, system, loss_arr, values_arr),
    lb = 0.0, ub = 100.0)
    sol = solve(optprob, GradientDescent(); time_limit=60)

    sol.u, sol.minimum

    return loss_arr, values_arr
end

function loss_finitediff(x, p)
    prob, tsteps, truth, system, loss_arr, values_arr = p
    g_val = x[1]

    pmap = [system.s12_LIF1.g_max => g_val] 

    newprob = remake(prob; p = pmap)
    sol = solve(newprob, Tsit5(); saveat = tsteps, save_end=false)

    pred = sol(tsteps)[system.IF2.IF2.oneport.v]
    truth = truth[system.IF2.IF2.oneport.v]


    #println(length(pred), length(truth))

    append!(loss_arr, sum(mean((truth .- pred).^2)))
    append!(values_arr, g_val)

    println("The loss error: ", mean((truth .- pred).^2))
    println("Value of g_max: ", g_val)

    return mean((truth .- pred).^2)
end


function generate_groundtruth_system(ref_sol)
    @named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),20, 0.0))
    neurons = [
        MTKNeuralToolkit.build_LIF(inp;name=:IF1),
        MTKNeuralToolkit.build_LIF(name=:IF2)
    ]
    connections = Dict(
        (1, 2) => [(type=:LIF, weight=1.0)]
    )

    ground_sys = build_network(connections, neurons)

    ground_prob = ODEProblem(ground_sys, Pair[], (0.0, 200.0))

    ground_sol = solve(ground_prob, Tsit5(); saveat=ref_sol.t);

    return ground_sol
end

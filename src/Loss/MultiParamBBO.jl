function loss_bbo(x, p)
    prob, tsteps, truth_vec, param_idx, state_idx,
        loss_arr, neurons, ground_spike_times = p

    loss_val = lif_loss(prob, x, tsteps, param_idx, state_idx,
                        truth_vec, neurons, ground_spike_times)

    push!(loss_arr, loss_val)
    return loss_val
end

function MultiParamBBO(system, prob, ground_sol, ground_spike_times,
                               neurons, params, opt, epoch)
    
    p = Progress(epoch; desc="BBO Optimising...")


    callback = function(state, loss)
        next!(p)
        return false
    end 
    
    tsteps = 0.0:0.1:500.0

    p_array, params_idx, state_idx = get_parameters(prob, system, params, neurons)
    loss_arr = Float64[]

    truth_vec = get_truth_vectors(ground_sol, neurons, tsteps)

    p0 = Float64[p_array[x] for x in params_idx]

    if opt == "BBO"
        optfn = OptimizationFunction(loss_bbo)
        optprob = OptimizationProblem(optfn, p0,
            (prob, tsteps, truth_vec, params_idx, state_idx,
            loss_arr, neurons, ground_spike_times);
            lb = fill(0.0, length(p0)),
            ub = fill(20.0, length(p0)))
        sol = solve(optprob, BBO_adaptive_de_rand_1_bin_radiuslimited();
                    maxiters = epoch, callback=callback)

        param_syms = parameters(prob.f.sys)
        for (i, idx) in enumerate(params_idx)
            println("Weight $i → $(param_syms[idx]) → BBO: $(round(sol.minimizer[i], digits=3))")
        end
        return loss_arr, sol.minimizer

    end

    finish!(p)
    error("unknown optimiser $opt")
end
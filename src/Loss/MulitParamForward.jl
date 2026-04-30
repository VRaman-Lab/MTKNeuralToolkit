function loss_forward(x, p)
    prob, tsteps, truth_vec, param_idx, state_idx,
        loss_arr, neurons, ground_spike_times = p

    loss_val = lif_loss(prob, x, tsteps, param_idx, state_idx,
                        truth_vec, neurons, ground_spike_times)

    Zygote.@ignore_derivatives begin
        push!(loss_arr, ForwardDiff.value(loss_val))
    end
    return loss_val
end

function MultiParamForward(system, prob, ground_sol, ground_spike_times,
                               neurons, params, opt, epoch)
    tsteps = 0.0:0.1:500.0


    p = Progress(epoch; desc="BBO Optimising...")


    callback = function(state, loss)
        next!(p)
        return false
    end 


    p_array, params_idx, state_idx = get_parameters(prob, system, params, neurons)
    loss_arr = Float64[]

    ground_state_syms = unknowns(ground_sol.prob.f.sys)
    truth_vec = []
    for n in neurons
        nm = string(nameof(n))
        pattern = nm * "₊" * nm * "₊oneport₊v"
        gi = findfirst(s -> contains(string(s), pattern), ground_state_syms)
        if gi === nothing
            @warn "could not find $nm in ground system"
        else
            push!(truth_vec, ground_sol(tsteps)[gi, :])
        end
    end

    p0 = Float64[p_array[x] for x in params_idx]

    optfn   = OptimizationFunction(loss_forward, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfn, p0,
        (prob, tsteps, truth_vec, params_idx, state_idx,
         loss_arr, neurons, ground_spike_times),
        )

    if opt == "ADAM"
        opt_fn      = Optimisers.Adam(0.001)
        current_lr  = Ref(0.01)
        last_losses = Ref(Float64[])

        cb = (state, l) -> begin
            push!(last_losses[], l)

            if state.iter % 200 == 0 && state.iter > 0
                new_lr = if     l > 50.0;  0.005
                        elseif l > 10.0;  0.001
                        elseif l > 5.0;   0.0005
                        else              0.0001
                        end
                if new_lr != current_lr[]
                    current_lr[] = new_lr
                    Optimisers.adjust!(state.original, eta = new_lr)
                    println("Epoch $(state.iter): loss=$(round(l, digits=4)) → lr=$new_lr")
                end
            end

            if length(last_losses[]) >= 300
                window = last_losses[][end-299:end]
                rel    = (maximum(window) - minimum(window)) /
                         (abs(mean(window)) + 1e-10)
                if rel < 1e-4
                    println("Epoch $(state.iter): converged (rel Δloss=$rel)")
                    return true      #
                end
            end
            return false
        end

        sol = solve(optprob, opt_fn; maxiters = epoch, callback = callback)
        finish!(p)

        param_syms = parameters(prob.f.sys)
        for (i, idx) in enumerate(params_idx)
            println("Weight $i → $(param_syms[idx]) → BBO: $(round(sol.minimizer[i], digits=3))")
        end
        return loss_arr, sol.minimizer
    end 
    
end
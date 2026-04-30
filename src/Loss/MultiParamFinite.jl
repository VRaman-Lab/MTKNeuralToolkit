
"
This is all test code and is in the works (Ella dissertation)
"

function MultiParamFinite(system, prob, ground_sol, ground_spike_times,
                               neurons, params, opt, epoch)
    tsteps = 0.0:0.1:500.0

    p = Progress(epoch; desc="BBO Optimising...")


    callback = function(state, loss)
        next!(p)
        return false
    end 

    p_array, params_idx, state_idx = get_parameters(prob, system, params, neurons)
    loss_arr = Float64[]

    truth_vec = get_truth_vectors(ground_sol, neurons, tsteps)

    p0 = Float64[p_array[x] for x in params_idx]
    optfn = OptimizationFunction(loss_finite, Optimization.AutoZygote())

    optprob = OptimizationProblem(
        optfn, p0,
        (prob, tsteps, truth_vec, params_idx, state_idx, loss_arr, neurons, ground_spike_times), 
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
        return loss_arr, sol.minimizer
    end 
end

function loss_finite(x, p)
    prob, tsteps, truth_vec, param_idx, state_idx,
        loss_arr, neurons, ground_spike_times = p

    loss_val = lif_loss(prob, x, tsteps, param_idx, state_idx,
                        truth_vec, neurons, ground_spike_times)

    Zygote.@ignore_derivatives begin
        push!(loss_arr, loss_val)
    end
    return loss_val
end

function ChainRulesCore.rrule(::typeof(lif_loss), prob, p_flat, tsteps, param_idx, state_idx, truth, neurons, ground_spike_times)
    loss_val = lif_loss(prob, p_flat, tsteps, param_idx, state_idx, truth, neurons, ground_spike_times)
    
    _prob = prob
    _tsteps = tsteps
    _param_idx = param_idx
    _state_idx = state_idx
    _p_flat = copy(p_flat)
    _truth = truth
    _neurons = neurons
    _ground_spike_times = ground_spike_times

    function lif_loss_pullback(Δ)
            
        δ = unthunk(Δ)
        ε = 0.1
        ∂p = zeros(Float64, length(_p_flat))
        for i in eachindex(_p_flat)
            p_plus  = copy(_p_flat); p_plus[i]  += ε
            p_minus = copy(_p_flat); p_minus[i] -= ε
            loss_plus  = lif_loss(_prob, p_plus,  _tsteps, _param_idx, _state_idx, _truth, _neurons, _ground_spike_times)
            loss_minus = lif_loss(_prob, p_minus, _tsteps, _param_idx, _state_idx, _truth, _neurons, _ground_spike_times)
            ∂p[i] = δ * (loss_plus - loss_minus) / (2ε)
        end
    
    
        return (NoTangent(), NoTangent(), ∂p, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return loss_val, lif_loss_pullback
end

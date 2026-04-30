
"
This is all test code and is in the works (Ella dissertation)
"
safe_std(x) = length(x) > 1 && any(xi != x[1] for xi in x) ? std(x) : 0.0

function CudaZygote_test(system, prob, ground_sol, neurons, params, opt, epoch)
    tsteps = unique(ground_sol.t)

    p_array, params_idx, state_idx = get_parameters(prob, system, params, neurons)
    loss_arr = []

    ground_state_syms = unknowns(ground_sol.prob.f.sys)
    truth_vec = []
    for n in neurons
        neuron_name = string(nameof(n))
        pattern = neuron_name * "₊" * neuron_name * "₊oneport₊v"
        ground_idx = findfirst(s -> contains(string(s), pattern), ground_state_syms)
        if !isnothing(ground_idx)
            push!(truth_vec, ground_sol(tsteps)[ground_idx, :])
        else
            println("Warning: could not find $neuron_name in ground system")
        end
    end

    truth_vec = [cu(Float32.(v)) for v in truth_vec]
    p0 = Float32[p_array[x] for x in params_idx]

    optfn = OptimizationFunction(loss, Optimization.AutoZygote())

    optprob = OptimizationProblem(
        optfn, p0,
        (prob, tsteps, truth_vec, params_idx, state_idx, loss_arr, neurons), 
    )
    if opt == "ADAM"
        sol = solve(optprob, ADAM(0.001); maxiters=epoch)
    end
    return loss_arr
end

function loss_cuda(x, p)
    prob, tsteps, truth_vec, param_idx, state_idx, loss_arr, neurons = p
    loss_val = lif_loss_cuda(prob, x, tsteps, param_idx, state_idx, truth_vec, neurons)
    println("Loss: ", loss_val, " ", x)

    Zygote.ignore() do
        push!(loss_arr, loss_val)
    end
    return loss_val
end


function lif_loss_cuda(prob, p_flat, tsteps, param_idx, state_idx, truth_vec, neurons)
    p_tunable, replace_p, _ = canonicalize(Tunable(), prob.p)
    p_new = [i in param_idx ? p_flat[findfirst(==(i), param_idx)] : p_tunable[i]
             for i in eachindex(p_tunable)]
    
    newprob = remake(prob;
        p  = replace_p(Float32.(p_new)),
        u0 = Float32.(prob.u0))

    sol, spike_times = forward_callback_cuda(newprob, neurons, tsteps)

    total_loss = 0.0f0
    for (i, neuron_state_i) in enumerate(state_idx)
        pred  = Float32[sol(t)[neuron_state_i] for t in tsteps]
        truth = Array(truth_vec[i])

        mean_loss = abs2(mean(pred) - mean(truth))
        std_loss  = abs2(safe_std(pred) - safe_std(truth))

        total_loss += mean_loss
        total_loss += 0.1f0 * std_loss 
    end
    return total_loss
end

function get_parameters(prob, system, params, neurons)
    param_syms = parameters(prob.f.sys)
    
    p_array, _, _ = SciMLStructures.canonicalize(Tunable(), prob.p)
    p_array = collect(p_array)  

    params_idx = Int[]
    for p in params
        matches = findall(param_syms) do s
            sym_str = split(string(s), "(")[1]
            contains(sym_str, p)
        end
        println("'$p' → matched: ", param_syms[matches])
        append!(params_idx, matches)
    end

    state_idx , neuron_dict = get_neuron_states(prob, system, neurons)
    println("Total optimizable params: ", length(params_idx))
    return p_array, params_idx, state_idx
end


function forward_callback_cuda(prob, neurons, tsteps)
    cb, spike_times = make_spike_callback(prob, neurons)
    sol = solve(
        ensemble, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend());
        trajectories = 1,
        callback     = cb,
        saveat       = tsteps,
        dtmax        = minimum(diff(tsteps)),
        verbose      = false
    )
    return sol, spike_times
end

function ChainRulesCore.rrule(::typeof(lif_loss_cuda), prob, p_flat, tsteps, param_idx, state_idx, truth, neurons)
    loss_val = lif_loss_cuda(prob, p_flat, tsteps, param_idx, state_idx, truth, neurons)

    function lif_loss_cuda_pullback(Δ)
        δ  = unthunk(Δ)
        ε  = 0.01f0                         
        n  = length(p_flat)
        ∂p = zeros(Float32, n)               

        function prob_func(prob, i, repeat)
            p_new   = copy(p_flat)
            param_i = (i - 1) % n + 1
            p_new[param_i] += i <= n ? ε : -ε

            p_tunable, replace_p, _ = canonicalize(Tunable(), prob.p)
            full_p = Float32[j in param_idx ? p_new[findfirst(==(j), param_idx)] : p_tunable[j]
                             for j in eachindex(p_tunable)] 
            remake(prob;
                p  = replace_p(full_p),
                u0 = Float32.(prob.u0))      
        end

        ensemble = EnsembleProblem(prob; prob_func = prob_func)

        sols = solve(
            ensemble, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
            trajectories = 2n,
            saveat       = tsteps
        )

        ∂p = Float32[
            δ * (compute_loss_cuda(sols[i],   state_idx, truth, tsteps) -
                 compute_loss_cuda(sols[i+n], state_idx, truth, tsteps)) / (2ε)
            for i in 1:n
        ]

        return (NoTangent(), NoTangent(), ∂p, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return loss_val, lif_loss_cuda_pullback
end

function compute_loss_cuda(sol, state_idx, truth_vec, tsteps)
    total = 0.0f0
    for (i, idx) in enumerate(state_idx)
        pred  = Float32[sol(t)[idx] for t in tsteps]
        truth = Array(truth_vec[i])  # pull from GPU to CPU for scalar ops

        total += abs2(mean(pred) - mean(truth))
        total += 0.1f0 * abs2(safe_std(pred) - safe_std(truth))
    end
    return total
end
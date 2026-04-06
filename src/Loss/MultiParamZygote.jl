
"
This is all test code and is in the works (Ella dissertation)
"
safe_std(x) = length(x) > 1 && any(xi != x[1] for xi in x) ? std(x) : 0.0

function MulitParamZygote_test(system, prob, ground_sol, neurons, params, opt, epoch)
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

    p0 = [p_array[x] for x in params_idx]
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

function loss(x, p)
    prob, tsteps, truth_vec, param_idx, state_idx, loss_arr, neurons = p
    loss_val = lif_loss(prob, x, tsteps, param_idx, state_idx, truth_vec, neurons)
    println("Loss: ", loss_val, " ", x)

    Zygote.ignore() do
        push!(loss_arr, loss_val)
    end
    return loss_val
end


function lif_loss(prob, p_flat, tsteps, param_idx, state_idx, truth_vec, neurons)
    p_tunable, replace_p, _ = canonicalize(Tunable(), prob.p)
    p_new = [i in param_idx ? p_flat[findfirst(==(i), param_idx)] : p_tunable[i]
             for i in eachindex(p_tunable)]
    newprob = remake(prob; p = replace_p(p_new))

    sol, spike_times = forward_callback(newprob, neurons, tsteps)

    total_loss = 0.0
    for (i, neuron_state_i) in enumerate(state_idx)
        pred  = [sol(t)[neuron_state_i] for t in tsteps]
        truth = truth_vec[i]

        mean_loss = abs2(mean(pred) - mean(truth))
        std_loss  = abs2(safe_std(pred) - safe_std(truth))

        total_loss += mean_loss
        total_loss += 0.1 * std_loss 
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

function get_neuron_states(prob, system, neurons)
    state_syms = unknowns(prob.f.sys)
    final_arr = Int[]
    neuron_dict = Dict{String, Int}()  

    neuron_syms = [state_syms[findall(s -> contains(string(s), string(nameof(n)) * "₊oneport₊v"), state_syms)] for n in neurons]

    for (n, sym) in zip(neurons, neuron_syms)
        idx = variable_index(prob, sym[1])
        push!(final_arr, idx)
        name = string(nameof(n))
        neuron_dict[name] = idx
        println("$name → index $idx")
    end

    return final_arr, neuron_dict
end

function forward_callback(prob, neurons, tsteps)
    cb, spike_times = make_spike_callback(prob, neurons)
    sol = solve(prob, Tsit5(); 
                callback = cb,
                saveat = tsteps,
                dtmax = minimum(diff(tsteps)),
                verbose = false)
    return sol, spike_times
end

function ChainRulesCore.rrule(::typeof(lif_loss), prob, p_flat, tsteps, param_idx, state_idx, truth, neurons)
    loss_val = lif_loss(prob, p_flat, tsteps, param_idx, state_idx, truth, neurons)
    
    _prob = prob
    _tsteps = tsteps
    _param_idx = param_idx
    _state_idx = state_idx
    _p_flat = copy(p_flat)
    _truth = truth
    _neurons = neurons

    function lif_loss_pullback(Δ)
    δ = unthunk(Δ)
    ε = 0.1
    ∂p = zeros(Float64, length(_p_flat))
    for i in eachindex(_p_flat)
        p_plus  = copy(_p_flat); p_plus[i]  += ε
        p_minus = copy(_p_flat); p_minus[i] -= ε
        loss_plus  = lif_loss(_prob, p_plus,  _tsteps, _param_idx, _state_idx, _truth, _neurons)
        loss_minus = lif_loss(_prob, p_minus, _tsteps, _param_idx, _state_idx, _truth, _neurons)
        ∂p[i] = δ * (loss_plus - loss_minus) / (2ε)
    end
    
    
    return (NoTangent(), NoTangent(), ∂p, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
end

    return loss_val, lif_loss_pullback
end


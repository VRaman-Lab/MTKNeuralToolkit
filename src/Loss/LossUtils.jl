sigmoid(x) = 1 / (1 + exp(-x))
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

function get_truth_vectors(ground_sol, neurons, tsteps)
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
    return truth_vec
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

function lif_loss(prob, p_flat, tsteps, param_idx, state_idx,
                  truth_vec, neurons, ground_spike_times)

    is_ad = eltype(p_flat) <: ForwardDiff.Dual
        
    p_tunable, replace_p, _ = canonicalize(Tunable(), prob.p)
    p_tunable = collect(p_tunable)

    p_new = [i in param_idx ? p_flat[findfirst(==(i), param_idx)] : p_tunable[i]
             for i in eachindex(p_tunable)]

    newprob = remake(prob; p = replace_p(p_new))
    cb, _ = make_spike_callback(newprob, neurons, ad_compatible=is_ad)

    sol = solve(newprob, Tsit5();
                callback       = cb,
                saveat         = tsteps,
                abstol         = 1e-8,
                reltol         = 1e-6,
                verbose        = false,
                sensealg       = ForwardDiffSensitivity())

    total_loss = zero(eltype(p_flat))
    dt = tsteps[2] - tsteps[1]
    τ = 1.0
    α = dt / τ
    sol_matrix = sol(tsteps)
    for (i, neuron_state_i) in enumerate(state_idx)
        pred  = sol_matrix[neuron_state_i, :]
        truth = truth_vec[i]
        n = length(pred)

        pred_fwd  = similar(pred)
        truth_fwd = similar(truth, eltype(pred))
        pred_fwd[1]  = pred[1]
        truth_fwd[1] = truth[1]
        for k in 2:n
            pred_fwd[k]  = α * pred[k]  + (1 - α) * pred_fwd[k-1]
            truth_fwd[k] = α * truth[k] + (1 - α) * truth_fwd[k-1]
        end


        pred_bwd  = similar(pred)
        truth_bwd = similar(truth, eltype(pred))
        pred_bwd[n]  = pred[n]
        truth_bwd[n] = truth[n]
        for k in (n-1):-1:1
            pred_bwd[k]  = α * pred[k]  + (1 - α) * pred_bwd[k+1]
            truth_bwd[k] = α * truth[k] + (1 - α) * truth_bwd[k+1]
        end


        smooth_pred  = (pred_fwd  .+ pred_bwd)  ./ 2
        smooth_truth = (truth_fwd .+ truth_bwd) ./ 2

        total_loss += mean(abs2.(smooth_pred .- smooth_truth))
    end

    return total_loss
end
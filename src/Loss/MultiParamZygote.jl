
"
This is all test code and is in the works (Ella dissertation)
"

function MulitParamZygote_test(system, ref_sol, prob, neurons, params, opt, epoch)
    ground_sol = generate_groundtruth_system(ref_sol)
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
            println("Ground $neuron_name → idx=$ground_idx, mean=$(mean(truth_vec[end]))")
        else
            println("Warning: could not find $neuron_name in ground system")
        end
    end

    p0 = [p_array[x] for x in params_idx]
    optfn = OptimizationFunction(loss, Optimization.AutoZygote())

    optprob = OptimizationProblem(
        optfn, p0,
        (prob, tsteps, truth_vec, params_idx, state_idx, loss_arr),
    )
    if opt == "ADAM"
        sol = solve(optprob, ADAM(0.001); maxiters=epoch)
    end
    return loss_arr
end

function loss(x, p)
    prob, tsteps, truth_vec, param_idx, state_idx, loss_arr = p
    loss_val = lif_loss(prob, x, tsteps, param_idx, state_idx, truth_vec)
    println("Loss: ", loss_val, " ", x)

    Zygote.ignore() do
        push!(loss_arr, loss_val)
    end
    return loss_val
end


function lif_loss(prob, p_flat, tsteps, param_idx, state_idx, truth_vec)
    p_tunable, replace_p, _ = canonicalize(Tunable(), prob.p)
    p_new = [i in param_idx ? p_flat[findfirst(==(i), param_idx)] : p_tunable[i]
             for i in eachindex(p_tunable)]
    newprob = remake(prob; p = replace_p(p_new))
    sol = solve(newprob, Tsit5(); 
                saveat = tsteps,
                dtmax = minimum(diff(tsteps)),
                verbose = false)
    
    total_loss = 0.0
    for (i, neuron_state_i) in enumerate(state_idx)
        pred = [sol(t)[neuron_state_i] for t in tsteps]

        total_loss += abs2(mean(pred) - mean(truth_vec[i]))
        total_loss += 0.1 * abs2(std(pred) - std(truth_vec[i]))
    end
    return total_loss
end


function generate_groundtruth_system(ref_sol)
    @named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),10, 0.0))
    neurons = [
    build_LIF(inp;name=:IF1),
    build_LIF(;name=:IF2),
    build_LIF(;name=:IF3),
    build_LIF(;name=:IF4)
    ]
    connections = Dict(
        (1, 2) => [(type=:LIF, weight = 5.0)],
        (2, 3) => [(type =:LIF, weight = 2.0)],
        (3, 4) => [(type =:LIF, weight = 2.0)],
    )



    ground_sys = build_network(connections, neurons)

    ground_prob = ODEProblem(ground_sys, Pair[], (0.0, 200.0))

    ground_sol = solve(ground_prob, Tsit5(); saveat=ref_sol.t);

    return ground_sol
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



function ChainRulesCore.rrule(::typeof(lif_loss), prob, p_flat, tsteps, param_idx, state_idx, truth)
    loss_val = lif_loss(prob, p_flat, tsteps, param_idx, state_idx, truth)
    
    _prob = prob
    _tsteps = tsteps
    _param_idx = param_idx
    _state_idx = state_idx
    _p_flat = copy(p_flat)
    _truth = truth

    function lif_loss_pullback(Δ)
    δ = unthunk(Δ)
    ε = 0.1
    ∂p = zeros(Float64, length(_p_flat))
    for i in eachindex(_p_flat)
        p_plus  = copy(_p_flat); p_plus[i]  += ε
        p_minus = copy(_p_flat); p_minus[i] -= ε
        loss_plus  = lif_loss(_prob, p_plus,  _tsteps, _param_idx, _state_idx, _truth)
        loss_minus = lif_loss(_prob, p_minus, _tsteps, _param_idx, _state_idx, _truth)
        ∂p[i] = δ * (loss_plus - loss_minus) / (2ε)
    end
    
    
    return (NoTangent(), NoTangent(), ∂p, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
end

    return loss_val, lif_loss_pullback
end


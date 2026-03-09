
"
This is all test code and is in the works (Ella dissertation)
"

function MulitParamZygote_test(system, ref_sol, prob, params, opt, epoch)
    ground_sol = generate_groundtruth_system(ref_sol)
    tsteps = unique(ground_sol.t)
    p_array, params_idx, state_idx = get_parameters(prob, system, params)

    truth_vec = []
    for idx in state_idx
        push!(truth_vec, ground_sol(tsteps)[idx, :])
    end

    p0 = [p_array[x] for x in params_idx]
    optfn   = OptimizationFunction(loss, Optimization.AutoZygote())
    optprob = OptimizationProblem(
        optfn, p0,
        (prob, tsteps, truth_vec, params_idx, state_idx, params),  # no f_plain/cb
    )
    if opt == "ADAM"
        sol = solve(optprob, ADAM(0.01); epochs = epoch)
    end
    sol.u, sol
end

function loss(x, p)
    prob, tsteps, truth_vec, param_idx, state_idx, params = p
    loss_val = lif_loss(prob, x, tsteps, param_idx, state_idx, truth_vec[1])
    println("Loss: ", loss_val, " ", x)
    return loss_val
end


function lif_loss(prob, p_flat, tsteps, param_idx, state_idx, truth)
    p_tunable, replace_p, _ = canonicalize(Tunable(), prob.p)
    p_new = [i in param_idx ? p_flat[findfirst(==(i), param_idx)] : p_tunable[i]
             for i in eachindex(p_tunable)]
    newprob = remake(prob; p = replace_p(p_new))
    sol = solve(newprob, Tsit5(); 
                saveat = tsteps,
                dtmax = minimum(diff(tsteps)),
                verbose = false)

    pred =[sol(t)[state_idx[1]] for t in tsteps]
    return mean(abs2, pred .- truth)
end

function generate_groundtruth_system(ref_sol)
    @named inp = TimeVaryingFunction(f = t -> ifelse((t > 10) & (t < 20),20, 0.0))
    neurons = [
        MTKNeuralToolkit.build_LIF(inp;name=:IF1),
        MTKNeuralToolkit.build_LIF(;name=:IF2),
        MTKNeuralToolkit.build_LIF(;name=:IF3),
        MTKNeuralToolkit.build_LIF(;name=:IF4),
        MTKNeuralToolkit.build_LIF(;name=:IF5)
    ]
    connections = Dict(
    (1, 2) => [(type=:LIF, weight=5.0)],
    (1, 3) => [(type=:LIF, weight=5.0)],
    (1, 4) => [(type=:LIF, weight=5.0)],
    (2, 5) => [(type=:LIF, weight=5.0)],
    (3, 5) => [(type=:LIF, weight=5.0)],
    (4, 5) => [(type=:LIF, weight=5.0)]
)

    ground_sys = build_network(connections, neurons)

    ground_prob = ODEProblem(ground_sys, Pair[], (0.0, 200.0))

    ground_sol = solve(ground_prob, Tsit5(); saveat=ref_sol.t);

    return ground_sol
end

function get_parameters(prob, system, params)
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

    state_idx = [variable_index(prob, system.IF5.IF5.oneport.v)]
    println("Total optimizable params: ", length(params_idx))
    return p_array, params_idx, state_idx
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
        ε = 1e-5
        ∂p = zeros(Float64, length(_p_flat))
        for i in eachindex(_p_flat)
            p_plus  = copy(_p_flat); p_plus[i]  += ε
            p_minus = copy(_p_flat); p_minus[i] -= ε
            loss_plus  = lif_loss(_prob, p_plus,  _tsteps, _param_idx, _state_idx, _truth)
            loss_minus = lif_loss(_prob, p_minus, _tsteps, _param_idx, _state_idx, _truth)
            ∂p[i] = δ * (loss_plus - loss_minus) / (2ε)
        end
        return (NoTangent(), NoTangent(), ∂p, NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end

    return loss_val, lif_loss_pullback
end


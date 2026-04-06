function generate_groundtruth_system(neurons, connections, target_weights)

    ground_connections = Dict{Tuple{Int,Int}, Vector{@NamedTuple{type::Symbol, weight::Float64}}}()

    if length(neurons)-1 != length(target_weights)
        throw("Invalid Target weight size, Number of taregt weights should be equal to the number of synapses defined!")
    end 

    for (k, (edge, synapses)) in enumerate(sort(collect(connections), by=first))
        new_synapses = map(synapses) do syn 
            pairs = [f => (f == :weight ? target_weights[k] : getfield(syn, f))
            for f in propertynames(syn)]
        (; pairs...)
        end 
        ground_connections[edge] = new_synapses
    end

    ground_sys  = build_network(ground_connections, neurons)
    ground_prob = ODEProblem(ground_sys, Pair[], (0.0, 200.0))
    cb, spike_counts = make_spike_callback(ground_prob, neurons)
    ground_sol = solve(ground_prob, Tsit5(); callback=cb)

    return ground_sol, spike_counts
end
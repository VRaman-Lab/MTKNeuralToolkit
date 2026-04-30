function make_ground_truth(prob, neurons, target_weights::Vector{Float64}, tsteps; ad_sys=false)
    param_syms = parameters(prob.f.sys)
    p_tunable, replace_p, _ = SciMLStructures.canonicalize(Tunable(), prob.p)
    p_new = collect(p_tunable)
    
    g_max_idx = findall(s -> contains(string(s), "g_max"), param_syms)
    
    if length(target_weights) != length(g_max_idx)
        throw(ArgumentError("Expected $(length(g_max_idx)) weights, got $(length(target_weights))"))
    end
    
    # Assign each weight individually
    for (i, idx) in enumerate(g_max_idx)
        p_new[idx] = target_weights[i]
    end
    
    ground_prob = remake(prob; p = replace_p(p_new))
    cb, spike_times = make_spike_callback(ground_prob, neurons, ad_compatible=ad_sys)
    return solve(ground_prob, Tsit5(); 
                 callback=cb, saveat=tsteps, abstol=1e-8, reltol=1e-6), spike_times
end
using Symbolics: fixpoint_sub, SymbolicT, Num, isarraysymbolic
using ModelingToolkit: unknowns, parameters, equations, @named, System, t_nounits as t, isparameter, is_derivative, getname, full_equations, continuous_events, observed, inputs, ImperativeAffect
using ModelingToolkitStandardLibrary.Blocks: RealInput
using ModelingToolkitStandardLibrary.Electrical: Ground
using DataFrames: DataFrame
using MacroTools: postwalk, @capture


function build_network(cell::Cell, N::Int; synapse_connections=[], ground_inputs=true, name=:network)

    compiled_cell = mtkcompile(cell.sys, inputs=cell.inputs)

    all_eqs = Equation[]
    all_vars = SymbolicT[]
    all_params = SymbolicT[]
    all_systems = System[]
    all_defaults = Dict{Any, Any}()
    all_events = []
    final_network_inputs = SymbolicT[]
    
    nodes = DataFrame(cell_idx=Int[], comp_idx=Int[], V=Any[], I_ext=Any[], Ca=Any[])
    
    driven_keys = Set{Tuple{Int,Int}}()
    for conn in synapse_connections
        post_cell, post_comp = conn[3], conn[4]
        push!(driven_keys, (post_cell, post_comp))
    end

    for n_idx in 1:N
        eqs, vars, ps, sub, defaults, events = clone_compiled_cell(compiled_cell, n_idx)
        append!(all_eqs, eqs)
        append!(all_vars, vars)
        append!(all_params, ps)
        merge!(all_defaults, defaults)
        append!(all_events, events)
        
        for (c_idx, comp) in enumerate(cell.compartments)
            V_orig = find_compiled_var(compiled_cell, comp.interfaces.V)
            I_ext_orig = find_compiled_var(compiled_cell, comp.interfaces.I_ext)
            
            V_new = sub[V_orig]
            I_ext_new = sub[I_ext_orig]
            
            all_defaults[V_new] = comp.V_init
            
            Ca_new = nothing
            if haskey(comp.interfaces, :Ca)
                Ca_orig = find_compiled_var(compiled_cell, comp.interfaces.Ca)
                Ca_new = sub[Ca_orig]
            end
            
            push!(nodes, (cell_idx=n_idx, comp_idx=c_idx, V=V_new, I_ext=I_ext_new, Ca=Ca_new))
            
            if !((n_idx, c_idx) in driven_keys)
                if ground_inputs
                    push!(all_eqs, I_ext_new ~ 0.0)
                else
                    push!(final_network_inputs, I_ext_new)
                end
            end
        end
    end

    syn_currents = Dict{Tuple{Int, Int}, Vector{Any}}()
    for (s_idx, conn) in enumerate(synapse_connections)
        pre_cell, pre_comp, post_cell, post_comp, gen = conn
        syn = gen(name=Symbol(:syn_, s_idx))
        push!(all_systems, syn)

        V_pre = nodes[(nodes.cell_idx .== pre_cell) .& (nodes.comp_idx .== pre_comp), :V][1]
        V_post = nodes[(nodes.cell_idx .== post_cell) .& (nodes.comp_idx .== post_comp), :V][1]

        push!(all_eqs, syn.V_pre ~ V_pre)
        push!(all_eqs, syn.V_post ~ V_post)

        if hasproperty(syn, :Ca_pre_sense)
            Ca_pre = nodes[(nodes.cell_idx .== pre_cell) .& (nodes.comp_idx .== pre_comp), :Ca][1]
            if Ca_pre !== nothing
                push!(all_eqs, syn.Ca_pre_sense.u ~ Ca_pre)
            end
        end

        key = (post_cell, post_comp)
        haskey(syn_currents, key) || (syn_currents[key] = Any[])
        
        if hasproperty(syn, :I_syn)
            push!(syn_currents[key], syn.I_syn)
        else
            push!(syn_currents[key], (syn.V_post - syn.E_rev) * syn.s * syn.g_max)
        end
    end

    for (key, currents) in syn_currents
        I_ext = nodes[(nodes.cell_idx .== key[1]) .& (nodes.comp_idx .== key[2]), :I_ext][1]
        push!(all_eqs, I_ext ~ sum(currents))
    end

    net_sys = System(all_eqs, t, all_vars, all_params;
                     initial_conditions=all_defaults,
                     systems=all_systems,
                     continuous_events=all_events,
                     inputs=final_network_inputs,
                     name=name)

    return Network(net_sys, nodes, DataFrame(), final_network_inputs)
end

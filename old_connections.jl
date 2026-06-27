function build_floating_compartment(capacitor, channels; name=:compartment)
    @named injector = CurrentSource()
    @named axial_injector = CurrentSource() 
    @named ground = Ground()

    @variables I_axial(t) I_ext(t)=0.0
    vars = SymbolicT[I_axial, I_ext]
    
    eqs = Equation[]
    
    push!(eqs, connect(capacitor.n, ground.g))
    push!(eqs, connect(injector.n, ground.g))
    push!(eqs, connect(axial_injector.n, ground.g))
    for c in channels
        push!(eqs, connect(c.n, ground.g))
    end
    
    p_connections = System[capacitor, injector, axial_injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))
    
    all_systems = System[ground, capacitor, injector, axial_injector]
    append!(all_systems, channels)

    push!(eqs, axial_injector.I.u ~ I_axial)
    push!(eqs, injector.I.u ~ I_ext)
    
    push!(eqs, D(I_axial) ~ 0)
    push!(eqs, D(I_ext) ~ 0)
    
    sys = System(eqs, t, vars, SymbolicT[]; systems = all_systems, name)
    return sys, (V=capacitor.v, I_axial=I_axial, I_ext=I_ext)

end

function vectorize_and_connect(compartments::Vector{<:Tuple}, axial_connections, N::Int; drivers=[], name=:pop)
    all_eqs = Equation[]
    all_vars_set = Set{SymbolicT}()
    all_systems = System[]
    all_events = []
    
    compiled_comps = [(mtkcompile(c[1]), c[2]) for c in compartments]
    clone_states = Dict{Tuple{Int, Int}, NamedTuple}()
    
    # Helper to generate unique, flattened names preserving the full namespace
    function make_new_name(sym, c_idx, i)
        sym_str = string(sym)
        sym_str = Base.replace(sym_str, "(t)" => "")
        flat_name = Base.replace(sym_str, "₊" => "_")
        return Symbol(:c_, c_idx, :_, flat_name, :_, i)
    end

    for i in 1:N
        for (c_idx, (c, iface)) in enumerate(compiled_comps)
            local_sub = Dict{Any, Any}()
            V_new = nothing
            I_axial_new = nothing
            I_ext_new = nothing
            
            # 1. Substitute ALL unknowns
            for u in unknowns(c)
                new_name = make_new_name(u, c_idx, i)
                new_v = only(@variables $new_name(t))
                local_sub[u] = new_v
                push!(all_vars_set, new_v)
                
                # Match interfaces by checking the suffix of the variable string
                u_str = string(u)
                if endswith(u_str, string(iface.V))
                    V_new = new_v
                elseif endswith(u_str, string(iface.I_axial))
                    I_axial_new = new_v
                elseif endswith(u_str, string(iface.I_ext))
                    I_ext_new = new_v
                end
            end
            
            # 2. Copy and substitute equations
            for eq in full_equations(c)
                # Skip dummy derivatives for interfaces so they become algebraic
                if Symbolics.is_derivative(eq.lhs)
                    lhs_str = string(eq.lhs)
                    if occursin(string(iface.I_axial), lhs_str) || occursin(string(iface.I_ext), lhs_str)
                        continue
                    end
                end
                push!(all_eqs, fixpoint_sub(eq, local_sub))
            end
            
            clone_states[(c_idx, i)] = (V=V_new, I_axial=I_axial_new, I_ext=I_ext_new)
            
            # 3. Copy and substitute continuous events
            for event in continuous_events(c)
                if event isa Pair
                    root_eqs = event.first
                    affect = event.second
                    
                    new_root = [fixpoint_sub(eq, local_sub) for eq in root_eqs]
                    
                    if affect isa AbstractVector
                        new_affect = [fixpoint_sub(eq, local_sub) for eq in affect]
                        push!(all_events, new_root => new_affect)
                    elseif affect isa ModelingToolkit.ImperativeAffect
                        # Substitute modified and observed symbols inside ImperativeAffect
                        new_mod = NamedTuple{keys(affect.modified)}([fixpoint_sub(v, local_sub) for v in affect.modified])
                        new_obs = NamedTuple{keys(affect.observed)}([fixpoint_sub(v, local_sub) for v in affect.observed])
                        new_affect = ModelingToolkit.ImperativeAffect(affect.f, new_mod, new_obs, affect.ctx)
                        push!(all_events, new_root => new_affect)
                    end
                end
            end
        end
        
        # 4. Add axial coupling equations
        for conn in axial_connections
            pre_idx, post_idx, R_val = conn
            pre_state = clone_states[(pre_idx, i)]
            post_state = clone_states[(post_idx, i)]
            
            I_flow = (pre_state.V - post_state.V) / R_val
            
            push!(all_eqs, pre_state.I_axial ~ -I_flow)
            push!(all_eqs, post_state.I_axial ~ I_flow)
        end
    end

    # 5. Process Drivers
    driven_exts = Set{Any}()
    for driver in drivers
        if length(driver) == 2
            c_idx, gen = driver
            for i in 1:N
                stim_name = Symbol(:stim_, c_idx, :_, i)
                stim = gen(name=stim_name)
                push!(all_systems, stim)
                I_ext_target = clone_states[(c_idx, i)].I_ext
                push!(all_eqs, I_ext_target ~ stim.output.u)
                push!(driven_exts, I_ext_target)
            end
        elseif length(driver) == 3
            c_idx, i, gen = driver
            stim_name = Symbol(:stim_, c_idx, :_, i)
            stim = gen(name=stim_name)
            push!(all_systems, stim)
            I_ext_target = clone_states[(c_idx, i)].I_ext
            push!(all_eqs, I_ext_target ~ stim.output.u)
            push!(driven_exts, I_ext_target)
        else
            error("Invalid driver format. Use (comp_idx, gen) or (comp_idx, clone_idx, gen).")
        end
    end

    # 6. Ground undriven I_ext variables algebraically
    for i in 1:N
        for (c_idx, _) in enumerate(compiled_comps)
            I_ext_new = clone_states[(c_idx, i)].I_ext
            if !(I_ext_new in driven_exts)
                push!(all_eqs, I_ext_new ~ 0.0)
            end
        end
    end
    
    # 7. Collect parameters (deduplicated to avoid clashes)
    all_params = SymbolicT[]
    seen_params = Set{SymbolicT}()
    for (c, _) in compiled_comps
        for p in parameters(c)
            if !(p in seen_params)
                push!(seen_params, p)
                push!(all_params, p)
            end
        end
    end
    
    # 8. Collect initial conditions
    all_ics = Dict{Any, Any}()
    for (c, _) in compiled_comps
        for (k, v) in ModelingToolkit.initial_conditions(c)
            if isparameter(k) 
                all_ics[k] = v
            end
        end
    end
    
    @named vec_sys = System(all_eqs, t, collect(all_vars_set), all_params; 
                            initial_conditions=all_ics, 
                            systems=all_systems, 
                            continuous_events=all_events, 
                            name=name)
    return vec_sys
end

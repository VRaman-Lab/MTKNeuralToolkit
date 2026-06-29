using Symbolics: fixpoint_sub, SymbolicT, Num, isarraysymbolic
using ModelingToolkit: unknowns, parameters, equations, @named, System, t_nounits as t, isparameter, is_derivative, getname, full_equations, continuous_events, observed, inputs, ImperativeAffect
using ModelingToolkitStandardLibrary.Blocks: RealInput
using DataFrames: DataFrame

# =========================================================
# 1. STRUCT DEFINITIONS
# =========================================================

struct Compartment
    sys::System
    interfaces::NamedTuple
    V_init::Float64
end

struct Cell
    sys::System
    compartments::Vector{Compartment}
    inputs::Vector{Any}
end

struct Network
    sys::System
    nodes::DataFrame
    edges::DataFrame
    inputs::Vector{Any}
end

# =========================================================
# 2. COMPARTMENT & CELL BUILDERS
# =========================================================

function build_compartment(capacitor, channels; name=:compartment, V_init=-65.0)
    @named injector = CurrentSource()
    @named axial_injector = CurrentSource()
    @named p = Pin()
    @named n = Pin()
    @named ground = Ground()

    vars = SymbolicT[]
    eqs = Equation[]
    
    # 1. Connect ALL negative terminals and the ground pin in ONE statement
    n_pins = Any[capacitor.n, injector.n, axial_injector.n, n, ground.g]
    for c in channels
        push!(n_pins, c.n)
    end
    push!(eqs, connect(n_pins...))
    
    # 2. Connect all positive terminals of the components together
    p_connections = System[capacitor, injector, axial_injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))
    
    # 3. Expose boundary pins for acausal connections
    push!(eqs, connect(p, capacitor.p))
    
    all_systems = System[ground, capacitor, injector, axial_injector, p, n]
    append!(all_systems, channels)

    sys = System(eqs, t, vars, SymbolicT[]; 
                 systems = all_systems, 
                 initial_conditions = Dict(capacitor.v => V_init), 
                 name)
    
    # 4. Extract the properly namespaced state variables for the interfaces
    cap_name = nameof(capacitor)
    V_state = getproperty(sys, cap_name).v
    
    # Namespace all exposed variables so they are ready for parent systems
    interfaces = (
        V = V_state, 
        p_pin = getproperty(sys, nameof(p)), 
        n_pin = getproperty(sys, nameof(n)), 
        I_ext = getproperty(sys, nameof(injector)).I.u,
        I_axial = getproperty(sys, nameof(axial_injector)).I.u,
        cap_name=cap_name
    )
    return Compartment(sys, interfaces, V_init)
end

# Example for the future: A compartment with Calcium tracking
function build_calcium_compartment(capacitor, channels, ca_var; name=:ca_compartment, V_init=-65.0)
    # ... setup ...
    interfaces = (
        V = capacitor.v, 
        I_ext = injector.I.u, 
        I_axial = axial_injector.I.u,
        Ca = ca_var  # Custom interface added!
    )
    return Compartment(sys, interfaces, V_init)
end

function build_cell(compartments::Vector{Compartment}, axial_connections; drivers=[], ground_undriven=true, name=:cell)
    eqs = Equation[]
    all_systems = System[]
    driven_exts = Set{Int}()
    vars = SymbolicT[]
    cell_inputs = SymbolicT[]
    
    for (idx, comp) in enumerate(compartments)
        push!(all_systems, comp.sys)
    end
    
    for conn in axial_connections
        pre_idx, post_idx, R_val = conn
        # Use the namespaced state variables from interfaces
        V_pre = compartments[pre_idx].interfaces.V
        V_post = compartments[post_idx].interfaces.V
        
        I_ax_pre = compartments[pre_idx].interfaces.I_axial
        I_ax_post = compartments[post_idx].interfaces.I_axial
        
        I_flow = (V_pre - V_post) / R_val
        push!(eqs, I_ax_pre ~ -I_flow)
        push!(eqs, I_ax_post ~ I_flow)
    end
    
    for (target, stim) in drivers
        idx = target isa Int ? target : findfirst(==(target), compartments)
        push!(all_systems, stim)
        push!(eqs, compartments[idx].interfaces.I_ext ~ stim.output.u)
        push!(driven_exts, idx)
    end
    
    if ground_undriven
        for (idx, comp) in enumerate(compartments)
            if !(idx in driven_exts)
                push!(eqs, comp.interfaces.I_ext ~ 0.0)
            end
        end
    else
        for (idx, comp) in enumerate(compartments)
            if !(idx in driven_exts)
                push!(cell_inputs, comp.interfaces.I_ext)
            end
        end
    end
    
    # Ground the unused acausal boundary pins so the mathematical cell is balanced
    for comp in compartments
        push!(eqs, comp.interfaces.p_pin.i ~ 0.0)
        push!(eqs, comp.interfaces.n_pin.i ~ 0.0)
    end
    
    @named cell_sys = System(eqs, t, vars, SymbolicT[]; systems=all_systems, inputs=cell_inputs, name=name)
    return Cell(cell_sys, compartments, cell_inputs)
end


# =========================================================
# 3. NETWORK BUILDERS (Acausal & Cloned)
# =========================================================

function build_electrical_network(compartments::Vector{Compartment}, axial_connections=[], synapse_connections=[]; drivers=[], name=:network)
    N = length(compartments)
    eqs = Equation[]
    all_systems = System[]
    
    for comp in compartments
        push!(all_systems, comp.sys)
    end

    # 1. Tie all grounds together by connecting the n pins.
    if N > 1
        n_pins = [compartments[i].sys.n for i in 1:N]
        push!(eqs, connect(n_pins...))
    end

    driven_compartments = Set{Int}()
    connected_p_pins = Set{Int}()

    # 2. Setup Driving Stimuli
    for (target, stim) in drivers
        idx = target isa Compartment ? findfirst(==(target), compartments) : target
        push!(driven_compartments, idx)
        push!(all_systems, stim)
        push!(eqs, connect(stim.output, compartments[idx].sys.injector.I))
    end

    # 3. Axial Connections (Internal topology)
    for conn in axial_connections
        pre_target, post_target, gen = conn[1], conn[2], conn[3]
        pre_idx = pre_target isa Compartment ? findfirst(==(pre_target), compartments) : pre_target
        post_idx = post_target isa Compartment ? findfirst(==(post_target), compartments) : post_target
        
        ax_name = length(conn) == 4 ? conn[4] : Symbol(:axial_, pre_idx, :_, post_idx)
        ax = gen(name=ax_name)
        push!(all_systems, ax)
        
        push!(connected_p_pins, pre_idx)
        push!(connected_p_pins, post_idx)
        
        if hasproperty(ax, :p1)
            push!(eqs, connect(compartments[pre_idx].sys.p, ax.p1))
            push!(eqs, connect(compartments[post_idx].sys.p, ax.p2))
            push!(eqs, connect(compartments[pre_idx].sys.n, ax.n1))
            push!(eqs, connect(compartments[post_idx].sys.n, ax.n2))
        else
            push!(eqs, connect(compartments[pre_idx].sys.p, ax.p))
            push!(eqs, connect(compartments[post_idx].sys.p, ax.n))
        end
    end

    # 4. Synaptic Connections (External topology)
    for conn in synapse_connections
        pre_target, post_target, gen = conn[1], conn[2], conn[3]
        pre_idx = pre_target isa Compartment ? findfirst(==(pre_target), compartments) : pre_target
        post_idx = post_target isa Compartment ? findfirst(==(post_target), compartments) : post_target
        
        syn_name = length(conn) == 4 ? conn[4] : Symbol(:syn_, pre_idx, :_, post_idx)
        syn = gen(name=syn_name)
        push!(all_systems, syn)

        push!(connected_p_pins, pre_idx)
        push!(connected_p_pins, post_idx)

        push!(eqs, connect(compartments[pre_idx].sys.p, syn.p1))
        push!(eqs, connect(compartments[post_idx].sys.p, syn.p2))
        push!(eqs, connect(compartments[pre_idx].sys.n, syn.n1))
        push!(eqs, connect(compartments[post_idx].sys.n, syn.n2))

        if hasproperty(syn, :V_pre_sense)
            push!(eqs, syn.V_pre_sense.u ~ compartments[pre_idx].interfaces.V)
        end
        if hasproperty(syn, :V_post_sense)
            push!(eqs, syn.V_post_sense.u ~ compartments[post_idx].interfaces.V)
        end
        if hasproperty(syn, :Ca_pre_sense) && haskey(compartments[pre_idx].interfaces, :Ca)
            push!(eqs, syn.Ca_pre_sense.u ~ compartments[pre_idx].interfaces.Ca)
        end
    end

    # 5. Ground undriven injectors and unconnected pins cleanly
    for i in 1:N
        if !(i in driven_compartments)
            push!(eqs, compartments[i].sys.injector.I.u ~ 0.0)
        end
        
        push!(eqs, compartments[i].sys.axial_injector.I.u ~ 0.0)
        
        if !(i in connected_p_pins)
            push!(eqs, compartments[i].sys.p.i ~ 0.0)
        end
    end

    net_sys = System(eqs, t, SymbolicT[], SymbolicT[]; systems = all_systems, name = name)
    return Network(net_sys, DataFrame(), DataFrame(), SymbolicT[])
end

# =========================================================
# 4. GENERIC CAUSAL CLONING ENGINE
# =========================================================

function find_compiled_var(compiled_sys, orig_var)
    target_name = getname(orig_var)
    for u in unknowns(compiled_sys)
        getname(u) == target_name && return u
    end
    for p in parameters(compiled_sys)
        getname(p) == target_name && return p
    end
    for inp in inputs(compiled_sys)
        getname(inp) == target_name && return inp
    end
    for eq in observed(compiled_sys)
        getname(eq.lhs) == target_name && return eq.lhs
    end
    error("Could not find interface variable $orig_var in compiled system.")
end

function clone_compiled_cell(compiled_cell::System, n_idx::Int)
    sub = Dict{Any, Any}()
    new_sts = SymbolicT[]
    new_ps = SymbolicT[]
    new_eqs = Equation[]
    
    function make_new_var(orig, is_param)
        name = getname(orig)
        new_name = Symbol(:n_, n_idx, :_, name)
        if isarraysymbolic(orig)
            dims = size(orig)
            return is_param ? only(@parameters $new_name[dims...]) : only(@variables $new_name(t)[dims...])
        else
            return is_param ? only(@parameters $new_name) : only(@variables $new_name(t))
        end
    end

    # Helper to safely clone a variable, avoiding duplicates
    function clone_var(orig, is_param)
        if haskey(sub, orig)
            return sub[orig]
        end
        new_var = make_new_var(orig, is_param)
        sub[orig] = new_var
        if is_param
            push!(new_ps, new_var)
        else
            push!(new_sts, new_var)
        end
        return new_var
    end

    # 1. Clone unknowns, inputs, and parameters
    for u in unknowns(compiled_cell); clone_var(u, false); end
    for inp in inputs(compiled_cell); clone_var(inp, false); end
    for p in parameters(compiled_cell); clone_var(p, true); end

    # 2. Clone observed variables as unknowns and their equations
    for eq in observed(compiled_cell)
        new_lhs = clone_var(eq.lhs, false)  # This now correctly adds them to new_sts!
        push!(new_eqs, new_lhs ~ fixpoint_sub(eq.rhs, sub))
    end

    # 3. Substitute and collect main equations
    append!(new_eqs, [fixpoint_sub(eq, sub) for eq in full_equations(compiled_cell)])

    # 4. Defaults (Initial conditions and bindings)
    new_defaults = Dict{Any, Any}()
    for (k, v) in ModelingToolkit.initial_conditions(compiled_cell)
        if haskey(sub, k); new_defaults[sub[k]] = v; end
    end
    for (k, v) in ModelingToolkit.bindings(compiled_cell)
        if haskey(sub, k); new_defaults[sub[k]] = v; end
    end

    # 5. Events
    new_events = []
    for ev in continuous_events(compiled_cell)
        root = [fixpoint_sub(eq, sub) for eq in ev.first]
        if ev.second isa AbstractVector
            affect = [fixpoint_sub(eq, sub) for eq in ev.second]
        elseif ev.second isa ImperativeAffect
            new_mod = NamedTuple{keys(ev.second.modified)}([fixpoint_sub(v, sub) for v in ev.second.modified])
            new_obs = NamedTuple{keys(ev.second.observed)}([fixpoint_sub(v, sub) for v in ev.second.observed])
            affect = ImperativeAffect(ev.second.f, new_mod, new_obs, ev.second.ctx)
        end
        push!(new_events, root => affect)
    end

    return new_eqs, new_sts, new_ps, sub, new_defaults, new_events
end


"""
    build_network(cell::Cell, N::Int; synapse_connections=[], ground_inputs=true, name=:network)

Pre-compiles a cell once, then uses `rename` to instantiate N clones as subsystems.
MTK flattens them efficiently during network compilation, bypassing O(N) tearing.
"""
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



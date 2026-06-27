using Symbolics: fixpoint_sub, SymbolicT
using ModelingToolkit: unknowns, parameters, equations, @named, System, t_nounits as t, isparameter, is_derivative, getname, full_equations, continuous_events, observed, inputs
using ModelingToolkitStandardLibrary.Blocks: RealInput

 
"""
build_compartment: Constructs a single neural compartment (soma/dendrite).
If `stimulus_block` is provided, it drives the internal current injector.
If `open_injector=true`, the injector control input remains open for external wiring.
"""
function build_compartment(capacitor, channels; stimulus_block=nothing, name=:neuron)
    @named injector = CurrentSource()
    @named p = Pin()
    @named n = Pin()

    @variables V(t)  
    vars = SymbolicT[V]
    
    eqs = Equation[]
    
    # Connect capacitor to boundary pins
    push!(eqs, connect(capacitor.p, p))
    push!(eqs, connect(capacitor.n, n))
    push!(eqs, V ~ p.v) 
    
    # Connect all positive pins together
    p_connections = System[capacitor, injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # Connect all negative pins together
    n_connections = System[capacitor, injector]
    append!(n_connections, channels)
    push!(eqs, connect([sys.n for sys in n_connections]...))
    
    all_systems = System[p, n, capacitor, injector]
    append!(all_systems, channels)

    if stimulus_block !== nothing
        push!(eqs, connect(stimulus_block.output, injector.I))
        push!(all_systems, stimulus_block)
    end
    
    return System(eqs, t, vars, SymbolicT[]; systems = all_systems, name)
end

function build_synapse(gate, battery; name)
    @named pre_p  = Pin() # Pre-synaptic sensing active point
    @named pre_n  = Pin() # Pre-synaptic sensing reference point
    @named post_p = Pin() # Post-synaptic active injection point
    @named post_n = Pin() # Post-synaptic reference return point
    
    vars = SymbolicT[]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    eqs = Equation[]
    # 1. Voltage sensing path (Pre-synaptic side)
    push!(eqs, connect(pre_p, gate.p1))
    push!(eqs, connect(pre_n, gate.n1))

    # 2. Current injection path (Post-synaptic side)
    push!(eqs, connect(post_p, gate.p2))
    push!(eqs, connect(gate.n2, battery.p))
    push!(eqs, connect(battery.n, post_n))
    
    subsystems = System[pre_p, pre_n, post_p, post_n, gate, battery]
    
    return System(eqs, t, vars, params; systems = subsystems, initial_conditions, guesses, name)
end

function build_vectorized_network(neurons::Vector{System}, synapse_blocks::Vector{System}; drivers=[], name=:vec_net)
    N = length(neurons)
    eqs = Equation[]
    all_systems = System[]
    append!(all_systems, neurons)

    # 1. Accumulate synaptic currents using Julia expressions
    I_exprs = [Num(0.0) for _ in 1:N]

    for block in synapse_blocks
        push!(all_systems, block)
        for i in 1:N
            push!(eqs, block.V_vec[i] ~ neurons[i].V)
            I_exprs[i] = I_exprs[i] + block.I_inj[i]
        end
    end

    # 2. Accumulate external stimulus directly into I_exprs
    for (target, stim) in drivers
        idx = target isa System ? findfirst(==(target), neurons) : target
        push!(all_systems, stim)
        I_exprs[idx] = I_exprs[idx] + stim.output.u
    end

    # 3. Map the final accumulated current directly to the injectors
    # (No if/else branches, just one clean equation per neuron)
    for i in 1:N
        push!(eqs, neurons[i].injector.I.u ~ I_exprs[i])
    end

    return System(eqs, t, SymbolicT[], SymbolicT[]; systems=all_systems, name=name)
end

# Helper to find the stim output for a specific neuron index
function stim_output(drivers, idx)
    for (target, stim) in drivers
        if (target isa System ? findfirst(==(target), neurons) : target) == idx
            return stim.output.u
        end
    end
    return Num(0.0)
end 


function build_fully_vectorized_network(neuron_block::System, synapse_blocks::Vector{System}; drivers=[], name=:vec_net)
    N = size(neuron_block.V)[1]

    eqs = Equation[]
    all_systems = System[neuron_block]
    append!(all_systems, synapse_blocks)

    # 1. Accumulate currents safely using a loop to maintain a mutable Vector{Num}
    I_exprs = [Num(0.0) for _ in 1:N]

    for block in synapse_blocks
        push!(eqs, block.V_vec ~ neuron_block.V)
        for i in 1:N
            I_exprs[i] = I_exprs[i] + block.I_inj[i]
        end
    end

    # 2. Add drivers (safe because I_exprs is still a standard Julia Vector)
    for (target, stim) in drivers
         @assert target isa Int "Fully vectorized networks require integer indices for drivers, because they contain a single monolithic neuron block."
        push!(all_systems, stim)
        I_exprs[target] = I_exprs[target] + stim.output.u
    end

    # 3. Push the connection equations
    for i in 1:N
        push!(eqs, neuron_block.I_inj[i] ~ I_exprs[i])
    end

    return System(eqs, t, SymbolicT[], SymbolicT[]; systems=all_systems, name=name)
end

"""
    build_electrical_network(compartments, axial_connections, synapse_connections; drivers=[], name=:network)

Construct an explicit, acausal circuit network from a flat list of compartments and two edge lists. 
Designed for multi-compartment spatial models, biophysical networks, and mixed-domain synapses.

# Arguments

- `compartments::Vector{System}`: A flat list of all compartment systems generated via `build_compartment`. 
  Indices in the connection lists refer to positions in this array.

- `axial_connections::Vector{<:Tuple}`: Internal structural connections (e.g., soma to dendrite). 
  Schema: `(pre_idx, post_idx, generator [, name::Symbol])`
  * `generator`: A function taking a keyword `name` that returns a `OnePort` or `TwoPort` system. 
    The builder will efficiently route standard `OnePort` components (like a `Resistor`) if no `p1` pin is found.

- `synapse_connections::Vector{<:Tuple}`: External synaptic connections between compartments.
  Schema: `(pre_idx, post_idx, generator [, name::Symbol])`
  * `generator`: A function taking a keyword `name` that returns a synapse system. 
    Synapses must be `TwoPort` systems for the acausal electrical current path. If the synapse 
    requires absolute voltage or calcium sensing, it should expose `RealInput`s named `V_pre_sense`, 
    `V_post_sense`, or `Ca_pre_sense`, which the builder will automatically wire.

# Keywords

- `drivers::Vector{Tuple{Int, System}}`: Optional list of causal input blocks targeting specific compartment indices. 
  Un-driven injectors are automatically grounded to `0.0` without instantiating redundant subsystems.
- `name::Symbol`: The system identifier for the resulting network.

# Example

```julia
compartments = [soma1, dend1, soma2]

# Efficient OnePort resistor for axial resistance
axial = [(1, 2, (; name) -> Resistor(R=0.5, name=name))]

# Mixed-domain synapse with absolute voltage sensing
synapses = [(2, 3, (; name) -> ChemicalSynapse(name=name, g_max=0.5))]

drivers = [(1, stim)] # Drive Soma1

net = build_electrical_network(compartments, axial, synapses; drivers=drivers)
```
"""

function build_electrical_network(compartments::Vector{System}, axial_connections, synapse_connections; drivers=[], name=:network)
N = length(compartments)
eqs = Equation[]
all_systems = System[]
append!(all_systems, compartments)

# 1. Single Global Ground
@named gnd = Ground()
push!(all_systems, gnd)
for i in 1:N
    push!(eqs, connect(compartments[i].n, gnd.g))
end

driven_compartments = Set{Int}()

# 2. Setup Driving Stimuli
for (target, stim) in drivers
    idx = target isa System ? findfirst(==(target), compartments) : target
    push!(driven_compartments, idx)
    push!(all_systems, stim)
    push!(eqs, connect(stim.output, compartments[idx].injector.I))
end

# 3. Axial Connections (Internal topology)
for conn in axial_connections
    pre_target, post_target, gen = conn[1], conn[2], conn[3]
    pre_idx = pre_target isa System ? findfirst(==(pre_target), compartments) : pre_target
    post_idx = post_target isa System ? findfirst(==(post_target), compartments) : post_target
    
    ax_name = length(conn) == 4 ? conn[4] : Symbol(:axial_, pre_idx, :_, post_idx)
    ax = gen(name=ax_name)
    push!(all_systems, ax)
    
    # Efficiently route OnePort vs TwoPort components
    if hasproperty(ax, :p1)
        push!(eqs, connect(compartments[pre_idx].p, ax.p1))
        push!(eqs, connect(compartments[post_idx].p, ax.p2))
        push!(eqs, connect(compartments[pre_idx].n, ax.n1))
        push!(eqs, connect(compartments[post_idx].n, ax.n2))
    else
        push!(eqs, connect(compartments[pre_idx].p, ax.p))
        push!(eqs, connect(compartments[post_idx].p, ax.n))
    end
end

# 4. Synaptic Connections (External topology)
for conn in synapse_connections
    pre_target, post_target, gen = conn[1], conn[2], conn[3]
    pre_idx = pre_target isa System ? findfirst(==(pre_target), compartments) : pre_target
    post_idx = post_target isa System ? findfirst(==(post_target), compartments) : post_target
    
    syn_name = length(conn) == 4 ? conn[4] : Symbol(:syn_, pre_idx, :_, post_idx)
    syn = gen(name=syn_name)
    push!(all_systems, syn)

    # Acausal wiring for synaptic currents
    push!(eqs, connect(compartments[pre_idx].p, syn.p1))
    push!(eqs, connect(compartments[post_idx].p, syn.p2))
    push!(eqs, connect(compartments[pre_idx].n, syn.n1))
    push!(eqs, connect(compartments[post_idx].n, syn.n2))

    # Auto-wiring for mixed-domain sensing
    if hasproperty(syn, :V_pre_sense)
        push!(eqs, syn.V_pre_sense.u ~ compartments[pre_idx].V)
    end
    if hasproperty(syn, :V_post_sense)
        push!(eqs, syn.V_post_sense.u ~ compartments[post_idx].V)
    end
    if hasproperty(syn, :Ca_pre_sense) && hasproperty(compartments[pre_idx], :Ca)
        push!(eqs, syn.Ca_pre_sense.u ~ compartments[pre_idx].Ca)
    end
end

# 5. Ground undriven injectors cleanly
for i in 1:N
    if !(i in driven_compartments)
        push!(eqs, compartments[i].injector.I.u ~ 0.0)
    end
end

return System(eqs, t, SymbolicT[], SymbolicT[]; systems = all_systems, name = name)
end

using Symbolics
using ModelingToolkit: unknowns, parameters, equations, @named, System, t_nounits as t

using ModelingToolkitStandardLibrary.Blocks: RealInput

using Symbolics: fixpoint_sub
using ModelingToolkit: unknowns, parameters, equations, @named, System, t_nounits as t, isparameter, is_derivative, getname

using ModelingToolkit: full_equations

struct Compartment
    sys::System
    interfaces::NamedTuple
end

function build_floating_compartment(capacitor, channels; name=:compartment)
    @named injector = CurrentSource()
    @named axial_injector = CurrentSource() 
    @named ground = Ground()

    @variables V(t)
    vars = SymbolicT[V]
    
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

    push!(eqs, V ~ capacitor.v)
    
    sys = System(eqs, t, vars, SymbolicT[]; systems = all_systems, name)
    return Compartment(sys, (V=V, cap_name=nameof(capacitor), I_axial=axial_injector.I.u, I_ext=injector.I.u))
end




struct Cell
    sys::System
    compartments::Vector{Compartment}
    inputs::Vector{Any}  # <-- ADD THIS
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
        V_pre = compartments[pre_idx].sys.V
        V_post = compartments[post_idx].sys.V
        I_ax_pre = compartments[pre_idx].sys.axial_injector.I.u
        I_ax_post = compartments[post_idx].sys.axial_injector.I.u
        
        I_flow = (V_pre - V_post) / R_val
        push!(eqs, I_ax_pre ~ -I_flow)
        push!(eqs, I_ax_post ~ I_flow)
    end

    
    for (target, stim) in drivers
        idx = target isa Int ? target : findfirst(==(target), compartments)
        push!(all_systems, stim)
        push!(eqs, compartments[idx].sys.injector.I.u ~ stim.output.u)
        push!(driven_exts, idx)
    end
    
    if ground_undriven
        for (idx, comp) in enumerate(compartments)
            if !(idx in driven_exts)
                push!(eqs, comp.sys.injector.I.u ~ 0.0)
            end
        end
    else
        # Network mode: directly expose the native RealInput of the CurrentSource
        for (idx, comp) in enumerate(compartments)
            if !(idx in driven_exts)
                push!(cell_inputs, comp.sys.injector.I.u)
            end
        end
    end
    
    @named cell_sys = System(eqs, t, vars, SymbolicT[]; systems=all_systems, inputs=cell_inputs, name=name)
    return Cell(cell_sys, compartments, cell_inputs)
end







struct Network
    sys::System
    nodes::DataFrame
    edges::DataFrame
    inputs::Vector{Any}
end

"""
    build_network(cell::Cell, N::Int; synapse_connections=[], ground_inputs=true, name=:network)

Pre-compile a cell once, then clone its simplified equations N times into a flat
network. This avoids re-running MTK's structural simplification on N identical cells,
trading hierarchical structure for compile-time speed on large homogeneous networks.

# Arguments
- `cell::Cell`: A cell built via `build_cell` with `ground_undriven=false`.

# Keywords
- `synapse_connections::Vector{Tuple}`: Each tuple is `(pre_cell, pre_comp, post_cell, post_comp, generator)`
  where `generator` is a function `(; name) -> System` producing a synapse with `V_pre`, `V_post`,
  `I_syn`, `s`, `E_rev`, and `g_max` variables/parameters.
- `ground_inputs::Bool`: If true, unconnected injector inputs are grounded to 0.0.
- `name::Symbol`: Network system name.

# Returns
- `Network`: The uncompiled flat system, a `nodes` DataFrame, and exposed inputs.
"""
function build_network(cell::Cell, N::Int; synapse_connections=[], ground_inputs=true, name=:network)
    # 1. Pre-compile the cell once — structural simplification runs here, not N times
    compiled_cell = mtkcompile(cell.sys, inputs=cell.inputs)

    all_eqs = Equation[]
    all_vars = SymbolicT[]
    all_params = SymbolicT[]
    all_defaults = Dict{Any, Any}()
    all_systems = System[]
    all_events = []
    nodes = DataFrame(cell_idx=Int[], comp_idx=Int[], V=Any[], I_ext=Any[])

    # Helper: create a renamed clone of a compiled symbolic variable
    function clone_sym(sym, n_idx, is_param)
        sym_str = Base.replace(string(sym), "(t)" => "")
        flat_name = Base.replace(sym_str, "₊" => "_")
        prefix = is_param ? :p_ : :n_
        new_name = Symbol(prefix, n_idx, :_, flat_name)
        return is_param ? only(@parameters $new_name) : only(@variables $new_name(t))
    end

    # Pre-compute lookup strings (avoid rebuilding inside the N-loop)
    compiled_inputs = ModelingToolkit.inputs(compiled_cell)
    input_strs = [Base.replace(string(inp), "(t)" => "") for inp in compiled_inputs]

    comp_lookup = Dict{Int, NamedTuple}()
    for (c_idx, comp) in enumerate(cell.compartments)
        comp_name = string(nameof(comp.sys))
        comp_lookup[c_idx] = (
            I_ext_str = "$(comp_name)₊injector₊I₊u(t)",
        )
    end

    # 2. Clone the compiled cell N times
    for n_idx in 1:N
        local_sub = Dict{Any, Any}()

        # Clone unknowns
        for u in unknowns(compiled_cell)
            new_v = clone_sym(u, n_idx, false)
            local_sub[u] = new_v
            push!(all_vars, new_v)
        end

        # Clone inputs as unknowns (they need to be driven by network equations)
        for inp in compiled_inputs
            haskey(local_sub, inp) && continue  # Already cloned as unknown
            new_v = clone_sym(inp, n_idx, false)
            local_sub[inp] = new_v
            push!(all_vars, new_v)
        end

        # Clone parameters (skip inputs — strip (t) from both sides for matching)
        for p in parameters(compiled_cell)
            p_str_clean = Base.replace(string(p), "(t)" => "")
            any(endswith.(p_str_clean, input_strs)) && continue
            new_p = clone_sym(p, n_idx, true)
            local_sub[p] = new_p
            push!(all_params, new_p)
        end

        # Clone initial conditions
        for (orig_var, val) in ModelingToolkit.initial_conditions(compiled_cell)
            if haskey(local_sub, orig_var)
                all_defaults[local_sub[orig_var]] = val
            end
        end

        # Clone equations (already simplified by the pre-compilation)
        for eq in full_equations(compiled_cell)
            push!(all_eqs, fixpoint_sub(eq, local_sub))
        end

        # Resolve V and I_ext for each compartment in this clone
        for (c_idx, comp) in enumerate(cell.compartments)
            comp_name = string(nameof(comp.sys))
            lookup = comp_lookup[c_idx]

            # Find V: search for capacitor voltage unknown directly
            # After mtkcompile, V is eliminated through an observed chain:
            # V -> capacitor.v -> capacitor.p.v - capacitor.n.v -> capacitor.p.v
            cap_name = string(comp.interfaces.cap_name)
            V_new = nothing
            for u in unknowns(compiled_cell)
                u_str = string(u)
                if occursin("$(comp_name)₊$(cap_name)₊", u_str) && endswith(u_str, "v(t)")
                    V_new = local_sub[u]
                    break
                end
            end

            # Find I_ext in compiled inputs
            I_ext_new = nothing
            for inp in compiled_inputs
                if endswith(string(inp), lookup.I_ext_str)
                    I_ext_new = local_sub[inp]
                    break
                end
            end

            push!(nodes, (cell_idx=n_idx, comp_idx=c_idx, V=V_new, I_ext=I_ext_new))
        end

        # Clone continuous events
        for event in continuous_events(compiled_cell)
            event isa Pair || continue
            root_eqs, affect = event.first, event.second
            new_root = [fixpoint_sub(eq, local_sub) for eq in root_eqs]

            if affect isa AbstractVector
                push!(all_events, new_root => [fixpoint_sub(eq, local_sub) for eq in affect])
            elseif affect isa ModelingToolkit.ImperativeAffect
                new_mod = NamedTuple{keys(affect.modified)}([fixpoint_sub(v, local_sub) for v in affect.modified])
                new_obs = NamedTuple{keys(affect.observed)}([fixpoint_sub(v, local_sub) for v in affect.observed])
                push!(all_events, new_root => ModelingToolkit.ImperativeAffect(affect.f, new_mod, new_obs, affect.ctx))
            end
        end
    end

    # 3. Wire up synapses
    syn_currents = Dict{Tuple{Int, Int}, Vector{Any}}()
    for (s_idx, conn) in enumerate(synapse_connections)
        pre_cell, pre_comp, post_cell, post_comp, gen = conn
        syn = gen(name=Symbol(:syn_, s_idx))
        push!(all_systems, syn)

        V_pre = nodes[(nodes.cell_idx .== pre_cell) .& (nodes.comp_idx .== pre_comp), :V][1]
        V_post = nodes[(nodes.cell_idx .== post_cell) .& (nodes.comp_idx .== post_comp), :V][1]

        push!(all_eqs, syn.V_pre ~ V_pre)
        push!(all_eqs, syn.V_post ~ V_post)

        key = (post_cell, post_comp)
        haskey(syn_currents, key) || (syn_currents[key] = Any[])
        # Inline the I_syn expression instead of referencing syn.I_syn variable.
        # This lets MTK cleanly eliminate I_syn as an observed variable
        # rather than keeping it as an algebraic unknown that confuses tearing.
        I_syn_expr = (syn.V_post - syn.E_rev) * syn.s * syn.g_max
        push!(syn_currents[key], I_syn_expr)
    end

    # 4. Ground or expose unconnected inputs
    final_inputs = SymbolicT[]
    for row in eachrow(nodes)
        key = (row.cell_idx, row.comp_idx)
        I_ext = row.I_ext

        if haskey(syn_currents, key)
            push!(all_eqs, I_ext ~ sum(syn_currents[key]))
        elseif ground_inputs
            push!(all_eqs, I_ext ~ 0.0)
        else
            push!(final_inputs, I_ext)
        end
    end

    net_sys = System(all_eqs, t, all_vars, all_params;
                     initial_conditions=all_defaults,
                     systems=all_systems,
                     continuous_events=all_events,
                     inputs=final_inputs,
                     name=name)

    return Network(net_sys, nodes, DataFrame(), final_inputs)
end














# ==========================================
# Core Network & Compartment Builders
# ==========================================

"""
    Compartment

A struct representing a single neural compartment (e.g., a soma, axon hillock, or dendritic segment).
It wraps the generated ModelingToolkit `System` along with metadata about its physical and 
electrical properties, and exposes a tuple of `interfaces` for acausal network connections.

# Fields
- `sys::System`: The underlying MTK system representing the compartment's equations.
- `interfaces::NamedTuple`: Exposed boundary variables and pins (e.g., `V`, `p_pin`, `n_pin`, `I_ext`, `I_syn`).
- `V_init::F`: The initial membrane voltage.
- `topology::Union{Scalar, Vectorized}`: The electrical topology of the compartment.
- `geometry::G`: The physical geometry used for scaling biophysical parameters.
- `morphology::M`: The spatial morphology used for rendering or spatial simulations.
"""
struct Compartment{M<:AbstractMorphology,G<:AbstractGeometry, F<:AbstractFloat}
    sys::System
    interfaces::NamedTuple
    V_init::F
    topology::Union{Scalar, Vectorized}
    geometry::G
    morphology::M
end

"""
    Network

A struct representing the complete assembled neural network. It encapsulates the fully 
connected MTK `System` and a vector of input variables for simulation drivers.

# Fields
- `sys::System`: The final compiled MTK system representing the entire network.
- `inputs::Vector{Any}`: A collection of symbolic input variables for external stimulation.
"""
struct Network
    sys::System
    inputs::Vector{Any}
end

"""
    SynapseSpec

A specification struct used to wire a synapse between a presynaptic voltage and a postsynaptic current.
It provides the mapping needed by `wire_synapses!` to inject currents into the correct compartments.

# Fields
- `pre_V`: The symbolic voltage variable of the presynaptic compartment.
- `post_V`: The symbolic voltage variable of the postsynaptic compartment.
- `post_I_syn`: The symbolic current variable of the postsynaptic compartment where the synapse will inject.
- `synapse::System`: The MTK synapse system component (e.g., `ExpSynapse`, `CholSynapse`).
- `post_comp::Union{Compartment, Nothing}`: The postsynaptic compartment struct (used for block synapse grounding logic).
"""
struct SynapseSpec
    pre_V
    post_V
    post_I_syn
    synapse
    post_comp::Union{Compartment, Nothing} 
end

"""
Outer constructor for `SynapseSpec` that defaults `post_comp` to `nothing`.
Useful for scalar synapses where block grounding logic is not required.
"""
SynapseSpec(pre_V, post_V, post_I_syn, synapse) = SynapseSpec(pre_V, post_V, post_I_syn, synapse, nothing)

"""
    CouplingSpec

A specification struct used to wire an acausal coupling (e.g., a Gap Junction) between two compartments.

# Fields
- `comp_i::Compartment`: The first compartment to be coupled.
- `comp_j::Compartment`: The second compartment to be coupled.
- `coupling::System`: The MTK coupling system component (e.g., `GapJunction`).
"""
struct CouplingSpec{C1<:Compartment, C2<:Compartment}
    comp_i::C1
    comp_j::C2
    coupling::System
end


# ==========================================
# Ion Configuration
# ==========================================

"""
    NoCalcium

A configuration struct indicating that a compartment has no Calcium dynamics.
When passed to `build_compartment`, it bypasses the creation of a `CalciumPool`.
"""
struct NoCalcium end

"""
    CalciumTracker

A configuration struct enabling Calcium dynamics within a compartment.
When passed to `build_compartment`, it instantiates a `CalciumPool` and connects it to all 
channels that expose a `ca_port`.

# Fields
- `decay::Union{Float64, Function}`: Either a time constant for linear decay, or a function 
  that takes the current Calcium concentration and returns the decay rate.
- `Ca_init::Float64`: The initial intracellular Calcium concentration.
"""
struct CalciumTracker
    decay::Union{Float64, Function}
    Ca_init::Float64
end

"""
Keyword argument constructor for `CalciumTracker`.
"""
CalciumTracker(; decay=100.0, Ca_init=0.0) = CalciumTracker(decay, Ca_init)


# ==========================================
# Internal Wiring Helpers
# ==========================================

"""
    wire_ions!(eqs, systems, channels, config, topology, name)

Internal helper function to wire ion dynamics into a compartment's equations and systems list.
Uses multiple dispatch to handle different ion configurations.

- If `config` is `NoCalcium`, it does nothing.
- If `config` is `CalciumTracker`, it creates a `CalciumPool` and connects it to all channels in the compartment that expose a `ca_port`.
"""
wire_ions!(eqs, systems, channels, ::NoCalcium, topology, name) = nothing
function wire_ions!(eqs, systems, channels, config::CalciumTracker, topology, name)
    # Pass decay to the CalciumPool
    ca_pool = CalciumPool(topology=topology, decay=config.decay, Ca_init=config.Ca_init, name=Symbol(name, :_ca_pool))
    push!(systems, ca_pool)
    
    ca_ports = System[ca_pool.port]
    for c in channels
        if hasproperty(c, :ca_port)
            push!(ca_ports, c.ca_port)
        end
    end
    push!(eqs, connect(ca_ports...))
end

"""
    wire_synapses!(eqs, systems, specs)

Internal helper function that wires a collection of `SynapseSpec`s into the network equations.
It binds the presynaptic and postsynaptic voltage variables to the synapse, and pre-collects 
convergent synapses by their target current variable to write a single summed equation per target.

Returns a tuple of `(driven_syn_targets, block_driven_targets)` used for grounding unconnected inputs.
"""
function wire_synapses!(eqs::Vector{Equation}, systems::Vector{System},
                        specs::Vector{SynapseSpec})
    syn_by_target = Dict{SymbolicT, Vector{SymbolicT}}()
    driven_syn_targets = Set{SymbolicT}()
    block_driven_targets = Set{SymbolicT}()

    for spec in specs
        push!(systems, spec.synapse)
        
        if hasproperty(spec.synapse, :V_pre)
            push!(eqs, spec.synapse.V_pre ~ spec.pre_V)
        end
        if hasproperty(spec.synapse, :V_post)
            push!(eqs, spec.synapse.V_post ~ spec.post_V)
        end

        key = spec.post_I_syn
        haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
        push!(syn_by_target[key], spec.synapse.I_syn)
        push!(driven_syn_targets, key)
        
        if spec.post_I_syn isa AbstractArray
            push!(block_driven_targets, spec.post_I_syn)
        end
    end

    for (target, currents) in syn_by_target
        if length(currents) == 1
            push!(eqs, target ~ currents[1])
        else
            push!(eqs, target ~ reduce(+, currents))
        end
    end

    return driven_syn_targets, block_driven_targets
end


# ==========================================
# Compartment & Network Builders
# ==========================================

"""
    build_compartment(capacitor, channels; name, V_init, topology, ion_config, geometry, morphology)

Builds a `Compartment` by connecting a `Capacitor`, current `injector`s, and a collection of ion `channels`.
This forms the fundamental electrical unit of a neuron. All positive terminals (p) are connected together 
to the membrane potential, and all negative terminals (n) are connected to ground. 

# Arguments
- `capacitor`: A `Capacitor` system defining the membrane capacitance.
- `channels`: A vector of ion channel systems (e.g., `GenericChannel`, `CaVChannel`).
- `name::Symbol`: The name of the compartment system.
- `V_init::Float64`: Initial membrane voltage (default -65.0 mV).
- `topology`: `Scalar()` or `Vectorized(N)` (default `Scalar()`).
- `ion_config`: `NoCalcium()` or `CalciumTracker()` to handle ion pools.
- `geometry`: Geometry struct for biophysical scaling (default `NoGeometry()`).
- `morphology`: Morphology struct for spatial data (default `NoMorphology()`).

# Returns
- A `Compartment` struct containing the assembled `System` and its exposed `interfaces`.
"""
function build_compartment(capacitor, channels; name=:compartment, V_init=-65.0, 
                           topology=Scalar(), ion_config=NoCalcium(), geometry = NoGeometry(), morphology=NoMorphology())
    
    p, n = create_pins(topology)
    injector, syn_injector = create_injectors(topology)
    init_v = init_voltage(topology, V_init)
    
    vars = SymbolicT[]
    eqs  = Equation[]

    # 1. Connect all negative terminals together
    n_pins = Any[capacitor.n, injector.n, syn_injector.n, n]
    for c in channels
        push!(n_pins, c.n)
    end
    push!(eqs, connect(n_pins...))

    # 2. Connect all positive terminals together
    p_connections = System[capacitor, injector, syn_injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # 3. Expose boundary pin for acausal connections (gap junctions)
    push!(eqs, connect(p, capacitor.p))

    all_systems = System[capacitor, injector, syn_injector, p, n]
    append!(all_systems, channels)

    # 4. Wire ions (dispatches on config and topology)
    wire_ions!(eqs, all_systems, channels, ion_config, topology, name)

    sys = System(eqs, t, vars, SymbolicT[];
                 systems = all_systems,
                 initial_conditions = Dict(capacitor.v => init_v),
                 name)

    cap_name = nameof(capacitor)
    V_state  = getproperty(sys, cap_name).v

    interfaces = (
        V       = V_state,
        p_pin   = getproperty(sys, nameof(p)),
        n_pin   = getproperty(sys, nameof(n)),
        I_ext   = getproperty(sys, nameof(injector)).I.u,
        I_syn   = getproperty(sys, nameof(syn_injector)).I.u,
        cap_name = cap_name
    )
    return Compartment(sys, interfaces, V_init, topology, geometry, morphology)
end

"""
    build_acausal_network(compartments; coupling_specs, synapse_specs, drivers, name)

Assembles a collection of `Compartment`s into a complete `Network` system. 
It handles grounding, wiring driving stimuli, gap junctions (via `CouplingSpec`), 
and chemical synapses (via `SynapseSpec`).

# Arguments
- `compartments::Vector{<:Compartment}`: The compartments making up the network.
- `coupling_specs`: A vector of `CouplingSpec` structs for acausal electrical connections.
- `synapse_specs`: A vector of `SynapseSpec` structs for directed chemical synapses.
- `drivers`: A vector of `(target, stim)` tuples, where `target` is a compartment or index, and `stim` is an MTK block, vector, or number.
- `name::Symbol`: The name of the overall network system.

# Returns
- A `Network` struct containing the assembled MTK `System`.
"""
function build_acausal_network(compartments::Vector{<:Compartment};
                                coupling_specs=CouplingSpec[],
                                synapse_specs=SynapseSpec[],
                                drivers=[],
                                name=:network)

    num_compartments = length(compartments)
    eqs = Equation[]
    all_systems = System[]

    for comp in compartments
        push!(all_systems, comp.sys)
    end

    driven_compartments = Set{Int}()
    gap_junctioned = Set{Int}()

    # 1. Ground each compartment individually (Dispatches on topology)
    for (i, comp) in enumerate(compartments)
        if haskey(comp.interfaces, :n_pin)
            gnd = create_ground(comp.topology, Symbol(:gnd_, i))
            push!(all_systems, gnd)
            push!(eqs, connect(gnd.g, comp.interfaces.n_pin))
        end
    end

    # 2. Driving stimuli
    for (target, stim) in drivers
        idx = target isa Compartment ? findfirst(==(target), compartments) : target
        push!(driven_compartments, idx)
        comp = compartments[idx]

        if haskey(comp.interfaces, :I_ext)
            if stim isa System
                push!(all_systems, stim)
                push!(eqs, comp.interfaces.I_ext ~ stim.output.u)
            elseif stim isa AbstractVector
                push!(eqs, comp.interfaces.I_ext ~ stim)
            elseif stim isa Number
                push!(eqs, comp.interfaces.I_ext ~ broadcast_stim(comp.topology, stim))
            end
        end
    end

    # 3. Ground undriven I_ext
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_ext) && !(i in driven_compartments)
            push!(eqs, comp.interfaces.I_ext ~ ground_current(comp.topology))
        end
    end

    # 4. Wire gap junctions via p_pin
    for (i, spec) in enumerate(coupling_specs)
        push!(all_systems, spec.coupling)
        
        if haskey(spec.comp_i.interfaces, :p_pin) && hasproperty(spec.coupling, :p1)
            push!(eqs, connect(spec.comp_i.interfaces.p_pin, spec.coupling.p1))
            push!(eqs, connect(spec.coupling.n1, spec.comp_i.interfaces.n_pin))
        end
        
        if haskey(spec.comp_j.interfaces, :p_pin) && hasproperty(spec.coupling, :p2)
            push!(eqs, connect(spec.comp_j.interfaces.p_pin, spec.coupling.p2))
            push!(eqs, connect(spec.coupling.n2, spec.comp_j.interfaces.n_pin))
        end
        
        push!(gap_junctioned, findfirst(==(spec.comp_i), compartments))
        push!(gap_junctioned, findfirst(==(spec.comp_j), compartments))
    end

    # 5. Identify block-synapsed compartments by index
    block_synapsed_compartments = Set{Int}()
    for spec in synapse_specs
        if spec.post_I_syn isa AbstractArray && spec.post_comp !== nothing
            idx = findfirst(==(spec.post_comp), compartments)
            if idx !== nothing
                push!(block_synapsed_compartments, idx)
            end
        end
    end

    # 6. Wire synapses
    driven_syn_targets, block_driven_targets = wire_synapses!(eqs, all_systems, synapse_specs)

    # 7. Ground non-synapsed I_syn (Dispatches on topology)
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_syn)
            if comp.interfaces.I_syn in block_driven_targets
                continue
            end
            ground_undriven_syn!(eqs, comp.topology, comp.interfaces.I_syn, driven_syn_targets)
        end
    end

    # 8. Ground non-gap-junctioned p_pin.i
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :p_pin) && !(i in gap_junctioned)
            push!(eqs, comp.interfaces.p_pin.i ~ ground_current(comp.topology))
        end
    end

    net_sys = System(eqs, t, SymbolicT[], SymbolicT[];
                     systems = all_systems, name = name)
                     
    return Network(net_sys, SymbolicT[])
end

"""
    build_synapse_block(pre_comp, post_comp, W; name, synapse_type, kwargs...)

Helper function to create a `SynapseSpec` using a vectorized synapse block.
It automatically determines the pre- and postsynaptic dimensions based on the weight matrix `W`
and binds it to the provided compartments.

# Arguments
- `pre_comp::Compartment`: The presynaptic compartment.
- `post_comp::Compartment`: The postsynaptic compartment.
- `W`: The weight matrix (dimensions `N_post` x `N_pre`).
- `name::Symbol`: The name for the synapse block system.
- `synapse_type`: The vectorized synapse component to use (defaults to `VectorizedExpSynapse`).
- `kwargs...`: Additional keyword arguments passed to `synapse_type` (e.g., `g_max`, `E_rev`).

# Returns
- A `SynapseSpec` configured for the network builder.
"""
function build_synapse_block(pre_comp, post_comp, W; name, 
                             synapse_type=VectorizedExpSynapse, kwargs...)
    N_pre  = size(W, 2)
    N_post = size(W, 1)
    syn = synapse_type(N_pre=N_pre, N_post=N_post, W=W; name=name, kwargs...)
    return SynapseSpec(pre_comp.interfaces.V, post_comp.interfaces.V,
                       post_comp.interfaces.I_syn, syn, post_comp)
end

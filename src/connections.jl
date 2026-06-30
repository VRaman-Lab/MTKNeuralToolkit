# connections.jl

using Symbolics: SymbolicT
using ModelingToolkit: t_nounits as t, connect, Equation, System, @named, getproperty, nameof

# =========================================================
# 1. STRUCT DEFINITIONS
# =========================================================

struct Compartment
    sys::System
    interfaces::NamedTuple
    V_init::Float64
    N::Union{Int, Nothing}  
end

struct Network
    sys::System
    nodes::DataFrame
    edges::DataFrame
    inputs::Vector{Any}
end

struct SynapseSpec
    pre_V
    post_V
    post_I_syn
    synapse
    post_comp::Union{Compartment, Nothing} 
end

# Backward-compatible constructor
SynapseSpec(pre_V, post_V, post_I_syn, synapse) = SynapseSpec(pre_V, post_V, post_I_syn, synapse, nothing)




struct CouplingSpec
    comp_i::Compartment
    comp_j::Compartment
    coupling::System
end


# =========================================================
# 2. COMPARTMENT & CELL BUILDERS
# =========================================================

function build_compartment(capacitor, channels; name=:compartment, V_init=-65.0, 
                           N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named injector  = CurrentSource()
        @named syn_injector = CurrentSource()
        @named p = Pin()
        @named n = Pin()
        init_v = V_init
    else
        @named injector  = CurrentSource(N=N)
        @named syn_injector = CurrentSource(N=N)
        @named p = VectorizedPin(N=N)
        @named n = VectorizedPin(N=N)
        init_v = fill(V_init, N)
    end

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
    return Compartment(sys, interfaces, V_init, N)
end






# =========================================================
# 3. SYNAPSE WIRING
# =========================================================

"""
    wire_synapses!(eqs, systems, specs)

Wires a collection of SynapseSpecs into the network equations.
Pre-collects convergent synapses by target and writes one sum equation per target.
Returns the set of driven I_syn targets (for grounding the rest).
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

        if spec.post_I_syn isa AbstractArray
            # Block synapse: equate arrays directly
            push!(eqs, spec.post_I_syn ~ spec.synapse.I_syn)
            push!(block_driven_targets, spec.post_I_syn) # Track the whole array!
            
            # Still add elements to driven_syn_targets for safety
            for i in 1:length(spec.post_I_syn)
                push!(driven_syn_targets, spec.post_I_syn[i])
            end
        else
            # Scalar synapse
            key = spec.post_I_syn
            haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
            push!(syn_by_target[key], spec.synapse.I_syn)
            push!(driven_syn_targets, key)
        end
    end

    for (target, currents) in syn_by_target
        push!(eqs, target ~ sum(currents))
    end

    return driven_syn_targets, block_driven_targets
end







# =========================================================
# 4. NETWORK BUILDER
# =========================================================

function build_acausal_network(compartments::Vector{Compartment};
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

    # 1. Ground each compartment individually
    for (i, comp) in enumerate(compartments)
        if haskey(comp.interfaces, :n_pin)
            gnd_name = Symbol(:gnd_, i)
            if isnothing(comp.N)
                gnd = Ground(name=gnd_name)
            else
                gnd = Ground(N=comp.N, name=gnd_name)
            end
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
                if isnothing(comp.N)
                    push!(eqs, comp.interfaces.I_ext ~ stim)
                else
                    push!(eqs, comp.interfaces.I_ext ~ fill(stim, comp.N))
                end
            end
        end
    end

    # 3. Ground undriven I_ext
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_ext) && !(i in driven_compartments)
            if isnothing(comp.N)
                push!(eqs, comp.interfaces.I_ext ~ 0.0)
            else
                push!(eqs, comp.interfaces.I_ext ~ zeros(Float64, comp.N))
            end
        end
    end

    # 4. Wire gap junctions via p_pin
    for (i, spec) in enumerate(coupling_specs)
        push!(all_systems, spec.coupling)
        
        # Defensive check for pins
        if haskey(spec.comp_i.interfaces, :p_pin) && hasproperty(spec.coupling, :p1)
            push!(eqs, connect(spec.comp_i.interfaces.p_pin, spec.coupling.p1))
            push!(eqs, connect(spec.coupling.n1, spec.comp_i.interfaces.n_pin))
        end
        
        if haskey(spec.comp_j.interfaces, :p_pin) && hasproperty(spec.coupling, :p2)
            push!(eqs, connect(spec.comp_j.interfaces.p_pin, spec.coupling.p2))
            push!(eqs, connect(spec.coupling.n2, spec.comp_j.interfaces.n_pin))
        end
        
        # Mark as gap junctioned so p_pin.i isn't grounded
        push!(gap_junctioned, findfirst(==(spec.comp_i), compartments))
        push!(gap_junctioned, findfirst(==(spec.comp_j), compartments))
    end

        # 5. Identify block-synapsed compartments by index using the Compartment object
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
    driven_syn_targets, _ = wire_synapses!(eqs, all_systems, synapse_specs)

    # 7. Ground non-synapsed I_syn
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :I_syn)
            # Skip entirely if driven by a block synapse (reliable integer check)
            if i in block_synapsed_compartments
                continue
            end
            
            if isnothing(comp.N)
                if !(comp.interfaces.I_syn in driven_syn_targets)
                    push!(eqs, comp.interfaces.I_syn ~ 0.0)
                end
            else
                for j in 1:comp.N
                    i_syn_j = comp.interfaces.I_syn[j]
                    if !(i_syn_j in driven_syn_targets)
                        push!(eqs, i_syn_j ~ 0.0)
                    end
                end
            end
        end
    end


    # 8. Ground non-gap-junctioned p_pin.i
    for i in 1:num_compartments
        comp = compartments[i]
        if haskey(comp.interfaces, :p_pin) && !(i in gap_junctioned)
            if isnothing(comp.N)
                push!(eqs, comp.interfaces.p_pin.i ~ 0.0)
            else
                push!(eqs, comp.interfaces.p_pin.i ~ zeros(Float64, comp.N))
            end
        end
    end

    net_sys = System(eqs, t, SymbolicT[], SymbolicT[];
                     systems = all_systems, name = name)
    return Network(net_sys, DataFrame(), DataFrame(), SymbolicT[])
end








function build_synapse_block(pre_comp, post_comp, W; name, 
                             synapse_type=VectorizedExpSynapse, kwargs...)
    N_pre  = size(W, 2)
    N_post = size(W, 1)
    syn = synapse_type(N_pre=N_pre, N_post=N_post, W=W; name=name, kwargs...)
    return SynapseSpec(pre_comp.interfaces.V, post_comp.interfaces.V,
                       post_comp.interfaces.I_syn, syn, post_comp) # Pass post_comp here
end



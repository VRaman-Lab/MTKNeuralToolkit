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

struct SynapseSpec
    pre_V::SymbolicT
    post_V::SymbolicT
    post_I_syn::SymbolicT
    synapse::System
end

# =========================================================
# 2. COMPARTMENT & CELL BUILDERS
# =========================================================

function build_compartment(capacitor, channels; name=:compartment, V_init=-65.0, 
                           N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named injector  = CurrentSource()
        @named syn_port  = SynapsePort()
        @named p = Pin()
        @named n = Pin()
        init_v = V_init
    else
        @named injector  = CurrentSource(N=N)
        @named syn_port  = SynapsePort(N=N)
        @named p = VectorizedPin(N=N)
        @named n = VectorizedPin(N=N)
        init_v = fill(V_init, N)
    end

    vars = SymbolicT[]
    eqs  = Equation[]

    # 1. Connect all negative terminals together
    n_pins = Any[capacitor.n, injector.n, n]
    for c in channels
        push!(n_pins, c.n)
    end
    push!(eqs, connect(n_pins...))

    # 2. Connect all positive terminals together (incl. syn_port)
    p_connections = System[capacitor, injector, syn_port]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # 3. Expose boundary pin for acausal connections (gap junctions)
    push!(eqs, connect(p, capacitor.p))

    all_systems = System[capacitor, injector, syn_port, p, n]
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
        I_syn   = getproperty(sys, nameof(syn_port)).I_syn,
        cap_name = cap_name
    )
    return Compartment(sys, interfaces, V_init)
end

function build_cell(compartments::Vector{Compartment}, axial_connections;
                    drivers=[], ground_undriven=true, name=:cell)
    eqs = Equation[]
    all_systems = System[]
    driven_exts = Set{Int}()
    gap_junctioned = Set{Int}()
    vars = SymbolicT[]
    cell_inputs = SymbolicT[]

    for comp in compartments
        push!(all_systems, comp.sys)
    end

    # 1. Connect all n pins together
    n_pins = [comp.interfaces.n_pin for comp in compartments]
    if length(n_pins) > 1
        push!(eqs, connect(n_pins...))
    end

    # 2. Global ground
    @named ground = Ground()
    push!(all_systems, ground)
    push!(eqs, connect(ground.g, compartments[1].interfaces.n_pin))

    # 3. Axial connections via GapJunction 
    for (i, conn) in enumerate(axial_connections)
        pre_idx, post_idx, R_val = conn
        gj = GapJunction(R=R_val, name=Symbol(:gj_, i))
        push!(all_systems, gj)

        # Connect gap junction between the two compartments
        push!(eqs, connect(compartments[pre_idx].interfaces.p_pin, gj.p1))
        push!(eqs, connect(gj.n1, compartments[pre_idx].interfaces.n_pin))
        push!(eqs, connect(compartments[post_idx].interfaces.p_pin, gj.p2))
        push!(eqs, connect(gj.n2, compartments[post_idx].interfaces.n_pin))

        push!(gap_junctioned, pre_idx, post_idx)
    end

    # 4. Drivers
    for (target, stim) in drivers
        idx = target isa Int ? target : findfirst(==(target), compartments)
        push!(driven_exts, idx)

        if stim isa System
            push!(all_systems, stim)
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim.output.u)
        elseif stim isa Number
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim)
        elseif stim isa AbstractVector
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim)
        end
    end

    # 5. Ground undriven I_ext
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

    # 6. Ground I_syn (no synapses at cell level)
    for comp in compartments
        push!(eqs, comp.interfaces.I_syn ~ 0.0)
    end

    # 7. Ground p_pin for non-gap-junctioned compartments
    for (idx, comp) in enumerate(compartments)
        if !(idx in gap_junctioned)
            push!(eqs, comp.interfaces.p_pin.i ~ 0.0)
        end
    end

    @named cell_sys = System(eqs, t, vars, SymbolicT[];
                             systems=all_systems,
                             inputs=cell_inputs,
                             name=name)
    return Cell(cell_sys, compartments, cell_inputs)
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

    for spec in specs
        push!(systems, spec.synapse)
        push!(eqs, spec.synapse.V_pre  ~ spec.pre_V)
        push!(eqs, spec.synapse.V_post ~ spec.post_V)

        if spec.post_I_syn isa AbstractArray
            # Block synapse: expand array to individual elements
            for i in 1:length(spec.post_I_syn)
                key = spec.post_I_syn[i]
                haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
                push!(syn_by_target[key], spec.synapse.I_syn[i])
            end
        else
            # Scalar synapse
            key = spec.post_I_syn
            haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
            push!(syn_by_target[key], spec.synapse.I_syn)
        end
    end

    for (target, currents) in syn_by_target
        push!(eqs, target ~ sum(currents))
    end

    return Set{SymbolicT}(keys(syn_by_target))
end



# =========================================================
# 4. NETWORK BUILDER
# =========================================================

function build_acausal_network(compartments::Vector{Compartment};
                                gap_junctions=[],
                                synapse_specs=SynapseSpec[],
                                drivers=[],
                                name=:network,
                                N::Union{Int, Nothing}=nothing)
    num_compartments = length(compartments)
    eqs = Equation[]
    all_systems = System[]

    for comp in compartments
        push!(all_systems, comp.sys)
    end

    # 1. Tie all grounds together
    n_pins = [compartments[i].interfaces.n_pin for i in 1:num_compartments]
    if length(n_pins) > 1
        push!(eqs, connect(n_pins...))
    end

    if isnothing(N)
        @named gnd = Ground()
    else
        @named gnd = Ground(N=N)
    end
    push!(all_systems, gnd)
    push!(eqs, connect(gnd.g, compartments[1].interfaces.n_pin))

    driven_compartments = Set{Int}()
    gap_junctioned = Set{Int}()

    # 2. Driving stimuli
    for (target, stim) in drivers
        idx = target isa Compartment ? findfirst(==(target), compartments) : target
        push!(driven_compartments, idx)

        if stim isa System
            push!(all_systems, stim)
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim.output.u)
        elseif stim isa AbstractVector
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim)
        elseif stim isa Number
            if isnothing(N)
                push!(eqs, compartments[idx].interfaces.I_ext ~ stim)
            else
                push!(eqs, compartments[idx].interfaces.I_ext ~ fill(stim, N))
            end
        end
    end

    # 3. Ground undriven I_ext
    for i in 1:num_compartments
        if !(i in driven_compartments)
            if isnothing(N)
                push!(eqs, compartments[i].interfaces.I_ext ~ 0.0)
            else
                push!(eqs, compartments[i].interfaces.I_ext ~ zeros(Float64, N))
            end
        end
    end

    # 4. Wire gap junctions via p_pin
    for (i, gj_spec) in enumerate(gap_junctions)
        comp_i, comp_j, R = gj_spec
        gj = GapJunction(R=R, name=Symbol(:gj_, i))
        push!(all_systems, gj)

        push!(eqs, connect(compartments[comp_i].interfaces.p_pin, gj.p1))
        push!(eqs, connect(gj.n1, compartments[comp_i].interfaces.n_pin))
        push!(eqs, connect(compartments[comp_j].interfaces.p_pin, gj.p2))
        push!(eqs, connect(gj.n2, compartments[comp_j].interfaces.n_pin))

        push!(gap_junctioned, comp_i, comp_j)
    end

    # 5. Wire synapses (pre-collects by target, handles convergence)
    driven_syn_targets = wire_synapses!(eqs, all_systems, synapse_specs)

    # 6. Ground non-synapsed I_syn
    for comp in compartments
        if isnothing(N)
            if !(comp.interfaces.I_syn in driven_syn_targets)
                push!(eqs, comp.interfaces.I_syn ~ 0.0)
            end
        else
            for i in 1:N
                i_syn_i = comp.interfaces.I_syn[i]
                if !(i_syn_i in driven_syn_targets)
                    push!(eqs, i_syn_i ~ 0.0)
                end
            end
        end
    end

    # 7. Ground non-gap-junctioned p_pin.i
    for i in 1:num_compartments
        if !(i in gap_junctioned)
            if isnothing(N)
                push!(eqs, compartments[i].interfaces.p_pin.i ~ 0.0)
            else
                push!(eqs, compartments[i].interfaces.p_pin.i ~ zeros(Float64, N))
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
                       post_comp.interfaces.I_syn, syn)
end


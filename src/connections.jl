"""
build_compartment: Constructs a single neural compartment (soma/dendrite).
If `stimulus_block` is provided, it drives the internal current injector.
If `open_injector=true`, the injector control input remains open for external wiring.
"""
function build_compartment(capacitor, channels; stimulus_block=nothing, name=:neuron)
    @named ground = Ground()
    @named injector = CurrentSource()
    @named p = Pin()
    @named n = Pin()

    @variables begin
        V(t)  
    end
    
    vars = SymbolicT[V]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()

    eqs = Equation[]
    push!(eqs, connect(capacitor.p, p))
    push!(eqs, connect(capacitor.n, n))
    push!(eqs, connect(capacitor.n, ground.g))
    push!(eqs, V ~ p.v) 
    
    # FIX: Use append! to combine scalars and vectors cleanly without syntax friction
    p_connections = System[capacitor, injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # FIX: Same clean combination pattern for the negative rail
    n_connections = System[capacitor, injector]
    append!(n_connections, channels)
    push!(eqs, connect([sys.n for sys in n_connections]...))
    
    # FIX: Safe subsystem collection construction
    all_systems = System[p, n, capacitor, ground, injector]
    append!(all_systems, channels)

    if stimulus_block !== nothing
        push!(eqs, connect(stimulus_block.output, injector.I))
        push!(all_systems, stimulus_block)
    end
    
    return System(eqs, t, vars, params; systems = all_systems, initial_conditions, guesses, name)
end




"""
build_channel: Factory function that wires a gating mechanism in series 
with an ionic reversal potential battery.
"""
function build_channel(gate, battery; name)
    # 1. Define clean, standardized boundary pins for the channel container
    @named p = Pin()
    @named n = Pin()

    eqs = Equation[]
    # Internal series connection between gate and battery
    push!(eqs, connect(gate.n, battery.p))
    
    # Connect the container's outer boundary pins to the internal elements
    push!(eqs, connect(p, gate.p))
    push!(eqs, connect(battery.n, n))
    
    subsystems = System[p, n, gate, battery]
    
    return System(
        eqs, 
        t, 
        SymbolicT[], 
        SymbolicT[]; 
        systems = subsystems, 
        name = name
    )
end


function EventSynapseGate(; name, g_max = 0.5, τ = 5.0, v_th = -20.0, w = 0.1)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport
    
    @parameters begin
        g_max = g_max
        τ = τ
        v_th = v_th
        w = w
    end
    params = SymbolicT[]
    push!(params, g_max)
    push!(params, τ)
    push!(params, v_th)
    push!(params, w)
    
    @variables begin
        s(t)
    end
    vars = SymbolicT[]
    push!(vars, s)
    
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    initial_conditions[s] = 0.0
    guesses = Dict{SymbolicT, SymbolicT}()
    
    eqs = Equation[]
    push!(eqs, i1 ~ 0.0)
    push!(eqs, D(s) ~ -s / τ)
    push!(eqs, i2 ~ v2 * s * g_max)
    
    root_eqs = Equation[]
    push!(root_eqs, v1 ~ v_th)
    
    affect = Equation[]
    push!(affect, s ~ Pre(s) + w)
    
    events = root_eqs => affect
    
    syn_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        initial_conditions, 
        guesses, 
        continuous_events = events,
        name
    )
    return extend(syn_sys, twoport)
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
 

"""
    build_electrical_network(neurons, connections; drivers=[], name=:neural_network)

Construct an explicit, acausal circuit network from a list of neurons and an edge list 
of synaptic blueprints. Ideal for biophysical models requiring physical current pathways 
(e.g., gap junctions, multi-compartment dynamics).

# Arguments
- `neurons::Vector{System}`: A flat list of compartment systems generated via `build_compartment`.
- `connections::Vector{<:Tuple}`: A 1D edge list of connections. Each tuple must follow 
  the schema: `(pre_idx, post_idx, synapse_blueprint, unique_name::Symbol)`.
  * `synapse_blueprint`: A functional factory (e.g., `name -> my_synapse(name)`) that 
    instantiates an isolated synapse system exposing `pre_p`, `pre_n`, `post_p`, and `post_n` pins.

# Keywords
- `drivers::Vector{Tuple{Int, System}}`: Optional list of causal input blocks (e.g. `Blocks.Sine`) 
  targeting specific neuron indices. Unbound injectors are automatically grounded inside the builder.
- `name::Symbol`: The system identifier for the resulting network macro-block.

# Composition Note (Factory List Pattern)
To build hierarchical or multi-population networks, do not nest network systems inside each other. 
Instead, write modular helper functions that return flat vectors of neurons and connection tuples, 
and combine them using `vcat` (e.g., `[nodes_A; nodes_B]`) before passing them to this builder.
"""
function build_electrical_network(neurons::Vector{System}, connections; drivers=[], name=:neural_network)
    # 1. Map system objects to their flat index positions
    neuron_to_idx = Dict(sys => i for (i, sys) in enumerate(neurons))
    
    eqs = Equation[]
    all_systems = System[]
    append!(all_systems, neurons) 

    # Track driven neurons for grounding logic
    driven_neurons = Set{Int}()
    for (target, stimulus_block) in drivers
        idx = target isa System ? neuron_to_idx[target] : target
        push!(driven_neurons, idx)
        push!(eqs, connect(stimulus_block.output, neurons[idx].injector.I))
        push!(all_systems, stimulus_block)
    end
    
    # 2. Wire the physical synapse blocks acausally
    for conn in connections
        # UNPACKING LOGIC:
        # If the user omitted a custom name, we auto-generate one on the fly!
        if length(conn) == 3
            pre_sys, post_sys, syn_generator = conn
            # E.g., :nrn1 and :nrn2 generates a fresh symbol :syn_nrn1_to_nrn2
            syn_name = Symbol(:syn_, nameof(pre_sys), :_to_, nameof(post_sys))
        else
            pre_sys, post_sys, syn_generator, syn_name = conn
        end
        
        pre_idx  = pre_sys  isa System ? neuron_to_idx[pre_sys]  : pre_sys
        post_idx = post_sys isa System ? neuron_to_idx[post_sys] : post_sys
    
        # Call the generator with the safe, unique name
        syn = syn_generator(name=syn_name)
        push!(all_systems, syn)

    
        # Acausal wiring using flat layout positions
        push!(eqs, connect(neurons[pre_idx].p, syn.p1))
        push!(eqs, connect(neurons[pre_idx].n, syn.n1))
        push!(eqs, connect(neurons[post_idx].injector.p, syn.p2))
        push!(eqs, connect(neurons[post_idx].injector.n, syn.n2)) 
    end
    
    # 3. Ground undriven injectors cleanly
    for i in eachindex(neurons)
        if !(i in driven_neurons)
            zero_block = Constant(k=0.0, name=Symbol(:zero_ground_, i))
            push!(all_systems, zero_block)
            push!(eqs, connect(zero_block.output, neurons[i].injector.I))
        end
    end
    
    return System(eqs, t, SymbolicT[], SymbolicT[]; systems = all_systems, name = name)
end

function build_factored_synapse_network(neuron_list::Vector{System}, connections::Vector{<:Tuple}; kwargs...)
    num_neurons = length(neuron_list)

    # Build a stable pointer lookup map
    neuron_to_idx = Dict(sys => i for (i, sys) in enumerate(neuron_list))

    # Safely normalize driver targets if they contain direct System objects
    clean_kwargs = Dict{Symbol, Any}(kwargs...)
    if haskey(clean_kwargs, :drivers)
        clean_kwargs[:drivers] = map(clean_kwargs[:drivers]) do (target, block)
            idx = target isa System ? neuron_to_idx[target] : target
            return (idx, block)
        end
    end

    # Normalize edges into the structural (i, j, spec, name) layout
    normalized_connections = map(connections) do c
        pre, post = c[1], c[2]
        spec      = c[3]

        i = pre  isa System ? neuron_to_idx[pre]  : pre
        j = post isa System ? neuron_to_idx[post] : post

        if length(c) == 3
            pre_name  = pre  isa System ? nameof(pre) : Symbol(:n, i)
            post_name = post isa System ? nameof(post) : Symbol(:n, j)
            syn_name  = Symbol(:synapse_, pre_name, :_to_, post_name)
            return (i, j, spec, syn_name)
        else
            return (i, j, spec, c[4])
        end
    end

    return _build_factored_synapse_network_impl(num_neurons, normalized_connections; clean_kwargs...)
end

function build_factored_synapse_network(neuron_list::Vector{System}, connectivity_matrix::Matrix; kwargs...)
    num_neurons = length(neuron_list)
    neuron_to_idx = Dict(sys => i for (i, sys) in enumerate(neuron_list))
    connections_list = Tuple[]

    for i in 1:num_neurons, j in 1:num_neurons
        spec = connectivity_matrix[i, j]
        (spec === nothing || i == j) && continue

        syn_name = Symbol(:synapse_, nameof(neuron_list[i]), :_to_, nameof(neuron_list[j]))
        push!(connections_list, (i, j, spec, syn_name))
    end

    # Normalize the driver targets for matrix inputs to prevent cross-dispatch bugs
    clean_kwargs = Dict{Symbol, Any}(kwargs...)
    if haskey(clean_kwargs, :drivers)
        clean_kwargs[:drivers] = map(clean_kwargs[:drivers]) do (target, block)
            idx = target isa System ? neuron_to_idx[target] : target
            return (idx, block)
        end
    end

    return _build_factored_synapse_network_impl(num_neurons, connections_list; clean_kwargs...)
end

function _build_factored_synapse_network_impl(num_neurons::Int, connections_list; drivers=[], name=:synapse_net)
    net_eqs = Equation[]
    subsystems = System[]

    # 1. Create causal IO boundaries using Array connectors (only 2 subsystems!)
    @named V_in = RealInputArray(nin = num_neurons)
    @named I_out = RealOutputArray(nout = num_neurons)

    push!(subsystems, V_in)
    push!(subsystems, I_out)

    # Array of arrays to collect current contributions per neuron
    current_contributions = [Num[] for _ in 1:num_neurons]

    # 2. Process connections from the unified list
    for (i, j, syn_constructor, syn_name) in connections_list
        # Instantiate the synapse component
        syn_instance = syn_constructor(name=syn_name)
        push!(subsystems, syn_instance)

        # Look up the boundary variables
        v_pre_var  = getproperty(syn_instance, :V_pre)
        v_post_var = getproperty(syn_instance, :V_post)
        i_syn_var  = getproperty(syn_instance, :I_syn)

        # Link synapse boundary vars directly to the array elements
        push!(net_eqs, v_pre_var ~ V_in.u[i])
        push!(net_eqs, v_post_var ~ V_in.u[j])

        # Collect the current variable expression for this neuron
        push!(current_contributions[j], i_syn_var)
    end

    # 3. Process external stimulus drivers
    for (neuron_idx, stimulus_block) in drivers
        push!(current_contributions[neuron_idx], -stimulus_block.output.u)
        push!(subsystems, stimulus_block)
    end

    # 4. Create intermediate sum variables and drive outputs
    @variables I_sum(t)[1:num_neurons]
    all_vars = collect(I_sum)

    for j in 1:num_neurons
        if isempty(current_contributions[j])
            push!(net_eqs, I_sum[j] ~ 0.0)
        else
            push!(net_eqs, I_sum[j] ~ sum(current_contributions[j]))
        end

        # Drive the array output element with the intermediate sum
        push!(net_eqs, I_out.u[j] ~ -I_sum[j])
    end

    return System(
        net_eqs, t, all_vars, SymbolicT[];
        systems = subsystems,
        name = name
    )
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

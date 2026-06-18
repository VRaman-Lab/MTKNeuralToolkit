"""
build_compartment: Constructs a single neural compartment (soma/dendrite).
If `stimulus_block` is provided, it drives the internal current injector.
If `open_injector=true`, the injector control input remains open for external wiring.
"""
function build_compartment(capacitor, channels; stimulus_block=nothing, open_injector=false, name=:neuron)
    @named ground = Ground()
    @named injector = CurrentSource()

    @named p = Pin()
    @named n = Pin()

    @variables begin
        V(t)  
    end
    vars = SymbolicT[V]
    params = SymbolicT[]
    guesses = Dict{SymbolicT, SymbolicT}()

    eqs = Equation[]
    push!(eqs, connect(capacitor.p, p))
    push!(eqs, connect(capacitor.n, n))
    push!(eqs, connect(capacitor.n, ground.g))
    
    # Positive rail connection (Parallel patch across the membrane)
    p_connections = System[capacitor.p]
    for ch in channels
        push!(p_connections, ch.p) # Uses the top-level channel pin refactored earlier
    end
    push!(p_connections, injector.p)
    push!(eqs, connect(p_connections...))

    # Negative rail connection (Reference return paths)
    n_connections = System[capacitor.n]
    for ch in channels
        push!(n_connections, ch.n) # Uses the top-level channel pin refactored earlier
    end
    push!(n_connections, injector.n)
    push!(eqs, connect(n_connections...))
    
    push!(eqs, V ~ p.v) 
    
    all_systems = System[p, n, capacitor, ground, injector]
    append!(all_systems, channels)

    # --- Clarified Input Routing Logic ---
    if stimulus_block !== nothing
        push!(eqs, connect(stimulus_block.output, injector.I))
        push!(all_systems, stimulus_block)
    elseif !open_injector
        # If the injector isn't actively driven or left open for network connections,
        # ground the causal input signal to 0.0 to balance the ODE equations.
        push!(eqs, injector.I.u ~ 0.0)
    end
    
    return System(eqs, t, vars, params; systems = all_systems, guesses, name)
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
build_electrical_network: Automatically maps and connects an arbitrary list of neurons,
synaptic pairs, and external drivers into a unified system.

NEED TO MAKE PRECOMPILATION FRIENDLY
"""
function build_electrical_network(neurons, connections; drivers=[], name=:neural_network)
    eqs = Equation[]
    all_systems = System[]
    append!(all_systems, neurons)
    
    vars = SymbolicT[]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    for (pre_idx, post_idx, gate, batt, syn_name) in connections
        # 1. Instantiate the synapse container
        syn = build_synapse(gate, batt; name=syn_name)
        push!(all_systems, syn)
        
        # 2. Pre-Synaptic Connection: Senses voltage across the pre-synaptic membrane
        push!(eqs, connect(neurons[pre_idx].p, syn.pre_p))
        push!(eqs, connect(neurons[pre_idx].n, syn.pre_n))
        
        # 3. Post-Synaptic Connection (Option B): Injects current directly 
        # into the post-synaptic neuron's internal injector terminal
        push!(eqs, connect(neurons[post_idx].injector.p, syn.post_p))
        push!(eqs, connect(neurons[post_idx].injector.n, syn.post_n))
    end
    
    # Handle external drivers (e.g., experimental driving currents)
    # Inside build_electrical_network, change the driver block loop:
    for (neuron_idx, stimulus_block) in drivers
        # Route the causal block output directly into the neuron's built-in injector signal
        push!(eqs, connect(stimulus_block.output, neurons[neuron_idx].injector.I))
        push!(all_systems, stimulus_block)
    end
    
    return System(eqs, t, vars, params; systems = all_systems, initial_conditions, guesses, name)
end



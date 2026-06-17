
using ModelingToolkit: renamespace




"""
build_neuron: Builder function that automatically compiles a parallel connection matrix
across a Soma and a hardcoded internal CurrentSource injector.
"""
function build_compartment(capacitor, channels; stimulus_block=nothing, name=:neuron)
    @named ground = Ground()
    @named injector = CurrentSource()

    @named p = Pin()
    @named n = Pin()

    @variables begin
        V(t)
    end
    vars = SymbolicT[]
    push!(vars, V)
    
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()

    eqs = Equation[]
    push!(eqs, connect(capacitor.p, p))
    push!(eqs, connect(capacitor.n, n))
    push!(eqs, connect(capacitor.n, ground.g))
    
    # Destructure connection arrays sequentially to avoid splatting types
    p_connections = System[]
    push!(p_connections, capacitor.p)
    for ch in channels
        push!(p_connections, ch.gate.p)
    end
    push!(p_connections, injector.p)
    push!(eqs, connect(p_connections...))

    n_connections = System[]
    push!(n_connections, capacitor.n)
    for ch in channels
        push!(n_connections, ch.batt.n)
    end
    push!(n_connections, injector.n)
    push!(eqs, connect(n_connections...))
    
    push!(eqs, V ~ p.v)
    
    # Assemble systems cleanly
    all_systems = System[]
    push!(all_systems, p)
    push!(all_systems, n)
    push!(all_systems, capacitor)
    push!(all_systems, ground)
    push!(all_systems, injector)
    append!(all_systems, channels)
    
    if stimulus_block !== nothing
        push!(eqs, connect(stimulus_block.output, injector.I))
        push!(all_systems, stimulus_block)
    else
        push!(eqs, injector.I.u ~ 0.0)
    end

    return System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = all_systems, 
        initial_conditions, 
        guesses, 
        name
    )
end



"""
build_channel: Factory function that wires a gating mechanism in series 
with an ionic reversal potential battery.
"""
function build_channel(gate, battery; name)
    eqs = Equation[]
    push!(eqs, connect(gate.n, battery.p))
    
    vars = SymbolicT[]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    subsystems = System[]
    push!(subsystems, gate)
    push!(subsystems, battery)
    
    return System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = subsystems, 
        initial_conditions, 
        guesses, 
        name
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
    @named p1 = Pin() # Post-synaptic active injection point
    @named p2 = Pin() # Post-synaptic reference return point
    
    vars = SymbolicT[]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    eqs = Equation[]
    # Complete the post-synaptic circuit through the boundary pins
    push!(eqs, connect(p1, gate.p2))
    push!(eqs, connect(gate.n2, battery.p))
    push!(eqs, connect(battery.n, p2)) # Clear path out to the neuron reference
    
    subsystems = System[]
    push!(subsystems, p1)
    push!(subsystems, p2)
    push!(subsystems, gate)
    push!(subsystems, battery)
    
    return System(eqs, t, vars, params; systems = subsystems, initial_conditions, guesses, name)
end
 


"""
build_network: Automatically maps and connects an arbitrary list of neurons,
synaptic pairs, and external drivers into a unified system.

NEED TO MAKE PRECOMPILATION FRIENDLY
"""
function build_network(neurons, connections; drivers=[], name=:neural_network)
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
        
        # 2. Extract the gate subsystem to access its raw pins directly
        inner_gate_name = ModelingToolkit.get_name(gate)
        inner_gate = getproperty(syn, inner_gate_name)
        
        # 3. Pre-Synaptic Connection (The trigger - draws 0 current)
        push!(eqs, connect(neurons[pre_idx].p, inner_gate.p1))
        push!(eqs, connect(neurons[pre_idx].n, inner_gate.n1))
        
        # 4. Post-Synaptic Connection (The closed loop injection)
        push!(eqs, connect(neurons[post_idx].p, syn.p1))
        push!(eqs, connect(neurons[post_idx].n, syn.p2)) # Completes the circuit loop!
    end
    
    # Handle drivers
    for (neuron_idx, stimulus_block, source_block) in drivers
        push!(eqs, connect(stimulus_block.output, source_block.I))
        push!(eqs, connect(source_block.p, neurons[neuron_idx].p))
        push!(eqs, connect(source_block.n, neurons[neuron_idx].n))
        push!(all_systems, stimulus_block)
        push!(all_systems, source_block)
    end
    
    return System(eqs, t, vars, params; systems = all_systems, initial_conditions, guesses, name)
end


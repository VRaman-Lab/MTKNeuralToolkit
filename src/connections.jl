
using ModelingToolkit: renamespace




"""
build_neuron: Builder function that automatically compiles a parallel connection matrix
across a Soma and a hardcoded internal CurrentSource injector.
"""
function build_compartment(capacitor, channels; stimulus_block=nothing, name=:neuron)
    @named ground = Ground()
    @named injector = CurrentSource() # Built-in injector source

    @named p = Pin()
    @named n = Pin()

    @variables V(t)
    
    eqs = [
        connect(capacitor.p, p)
        connect(capacitor.n, n)
        connect(capacitor.n, ground.g)
        
        # Parallel networks hook up the channels AND the built-in injector branch
        connect(capacitor.p, [ch.gate.p for ch in channels]..., injector.p)
        connect(capacitor.n, [ch.batt.n for ch in channels]..., injector.n)
        V ~ p.v
    ]
    
    all_systems = [p,n,capacitor, ground, injector, channels...]
    
    # Structural evaluation to ensure equation balancing
    if stimulus_block !== nothing
        # If an external block is passed, connect it to the inner injector's RealInput
        push!(eqs, connect(stimulus_block.output, injector.I))
        push!(all_systems, stimulus_block)
    else
        # If no block is passed, pin the injector to 0.0 to balance the system matrix
        push!(eqs, injector.I.u ~ 0.0)
    end

    
    
    return System(eqs, t, [V], []; name, systems = all_systems)
end



"""
build_channel: Factory function that wires a gating mechanism in series 
with an ionic reversal potential battery.
"""
function build_channel(gate, battery; name)
    eqs = [
        connect(gate.n, battery.p)
    ]
    return System(eqs, t, [], []; name, systems = [gate, battery])
end




function connect_synapse()
    missing
end


@component function EventSynapseGate(; name, g_max = 0.5, τ = 5.0, v_th = -20.0, w = 0.1)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport
    
    params = @parameters(
        g_max = g_max,
        τ = τ,
        v_th = v_th,
        w = w
    )
    vars = @variables s(t) = 0.0
    
    D = Differential(t)
    
    eqs = [
        i1 ~ 0.0,
        D(s) ~ -s / τ,
        i2 ~ v2 * s * g_max
    ]
    
    root_eqs = [v1 ~ v_th] 
    affect   = [s ~ Pre(s) + w]
    events   = root_eqs => affect
    
    base_sys = System(eqs, t, vars, [g_max, τ, v_th, w]; name, continuous_events = events)
    return extend(base_sys, twoport)
end

function build_synapse(gate, battery; name)
    # 1. Create clean, explicit surface boundary pins for the synapse container
    @named p1 = Pin()
    @named n1 = Pin()
    @named p2 = Pin()
    @named n2 = Pin()
    
    # 2. Map the boundary pins to the internal components, and wire them in series
    eqs = [
        connect(p1, gate.p1),
        connect(n1, gate.n1),
        
        connect(p2, gate.p2),
        connect(gate.n2, battery.p),
        connect(battery.n, n2)
    ]
    
    # 3. Use standard composition. This leaves the 'syn_gate' namespace completely intact!
    return System(eqs, t, [], []; name, systems = [p1, n1, p2, n2, gate, battery])
end
 
function neuron_connect(pre_compartment, post_compartment, synapse)
    return [
        connect(pre_compartment.p, synapse.p1),
        connect(pre_compartment.n, synapse.n1),
        connect(post_compartment.p, synapse.p2),
        connect(post_compartment.n, synapse.n2)
    ]
end

export neuron_connect

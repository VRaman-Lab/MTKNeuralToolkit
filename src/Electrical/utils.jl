using ModelingToolkit

function build_channel(gate, battery; name)
    @named oneport = OnePort()
    @unpack v, i = oneport

    # Acausal electrical rules for elements in series:
    # 1. The total voltage across the channel is the sum of the gate and battery voltages.
    # 2. The current passing through the gate, battery, and total channel is identical.
    eqs = [
        v ~ gate.v + battery.v,
        i ~ gate.i,
        gate.i ~ battery.i
    ]

    base_sys = System(eqs, t, [], []; name, systems = [gate, battery])
    return extend(base_sys, oneport)
end

"""
    build_neuron(neuron, input=Constant(k=0.0); channels)

Stitches channels and inputs onto a neuron body safely using duck-typing.
"""
function build_neuron(neuron, input=nothing; channels, name=nameof(neuron))
    connections = Pin[]
    
    # 1. Standard channel electrical routing
    for chan in channels
        push!(connections, connect(chan.p, neuron.oneport.p))
        push!(connections, connect(neuron.ground.g, neuron.oneport.n, chan.n))
        
        # 2. Automated Ionic Routing via Duck Typing
        # Eliminates the nested 'channel.conductance.ca.p' digging. 
        # If the channel exposes a top-level ionic port, connect it!
        if hasproperty(chan, :ca) && hasproperty(neuron, :ca)
            push!(connections, connect(chan.ca, neuron.ca))
        elseif hasproperty(chan, :conductance) && hasproperty(chan.conductance, :ca) && hasproperty(neuron, :ca)
            push!(connections, connect(chan.conductance.ca, neuron.ca))
        end
    end
    
    # 3. Optional input handling
    systems = Any[neuron; channels...]
    if input !== nothing
        push!(connections, connect(input.output, neuron.I))
        push!(systems, input)
    end
    
    return compose(ODESystem(connections, t; name), systems)
end

"""
    connect_synapse(synapse, pre_neuron, post_neuron; name)

Unifies synapse linking without needing unsafe 'nameof' property reflection loops.
"""
function connect_synapse(synapse, pre_neuron, post_neuron; name)
    # Pull the membrane terminals safely regardless of naming conventions
    pre_membrane  = hasproperty(pre_neuron, :oneport) ? pre_neuron.oneport.p : pre_neuron.p
    post_membrane = hasproperty(post_neuron, :oneport) ? post_neuron.oneport.p : post_neuron.p
    
    eqs = [
        connect(synapse.pre, pre_membrane),
        connect(synapse.post, post_membrane)
    ]
    
    return compose(ODESystem(eqs, t; name), [pre_neuron, post_neuron, synapse])
end


@named capacitor = Capacitor(C = 1.0)
@named sodium    = nagates(g = 120.0, E = 50.0)    # Na+ channel
@named potassium = kgates(g = 36.0, E = -77.0)     # K+ channel
@named leak      = lgates(g = 0.3, E = -54.4)      # Passive Leak

# A ConstantVoltage component can act as our command stimulus (e.g., holding potential)
@named stimulus  = ConstantVoltage(V = -65.0) 
@named ground    = Ground()

# --- 2. Define the Parallel Connections ---
# In a Hodgkin-Huxley model, the intracellular sides (positive pins '.p') 
# and extracellular sides (negative pins '.n') are all tied together.
hh_eqs = [
    # Connect all intracellular nodes together
    connect(capacitor.p, sodium.p, potassium.p, leak.p, stimulus.p)
    
    # Connect all extracellular nodes together
    connect(capacitor.n, sodium.n, potassium.n, leak.n, stimulus.n)
    
    # Anchor the extracellular space to Ground reference (0 mV)
    connect(capacitor.n, ground.g)
]

# --- 3. Build and Compile the Full Neuron System ---
@named hh_neuron = System(
    hh_eqs, 
    t; 
    systems = [capacitor, sodium, potassium, leak, stimulus, ground]
)

# Compile the structural model down into simplified mathematical equations
hh_compiled = mtkcompile(hh_neuron)


"""
    Build channel from gates: put in series with voltage.
"""
function build_channel(conductance, reversal;name)

    @named p = Pin()
    @named n = Pin()
    connections = [
        connect(conductance.p, reversal.n)
        connect(conductance.n, n)
        connect(reversal.p, p)
    ]
    return compose(ODESystem(connections, t; name), [p,n,conductance,reversal])
end

function build_channel_explicit(conductance, reversal;name)

    @named p = Pin()
    @named n = Pin()
    connections = [
        connect(conductance.oneport.p, reversal.n)
        connect(conductance.oneport.n, n)
        connect(reversal.p, p)
    ]
    return compose(ODESystem(connections, t; name), [p,n,conductance,reversal])
end

"""
    build_ca_channel(conductance; name)

Build a calcium channel with internal dynamic reversal potential.
The channel must have an internal Nernst equation for reversal calculation.
"""

function build_neuron(neuron, input; channels)
     channel_connections = [[
         connect(channel.p, neuron.p),
         connect(neuron.ground.g, neuron.n, channel.n)
     ] for channel in channels]

     input_connection = connect(input.output, neuron.I)
    calcium_flux_connections = [[
            connect(channel.conductance.ca.p, neuron.ca.p),
            connect(neuron.ca.n, channel.conductance.ca.n),          
     ] for channel in channels if hasproperty(channel.conductance, :ca) ]

     connections = vcat(channel_connections..., input_connection, calcium_flux_connections...)
     connected_system = compose(ODESystem(connections, t, name=nameof(neuron)), [channels..., neuron,input])
     return connected_system
end

function build_neuron(neuron; channels)
    channel_connections = [[
         connect(channel.p, neuron.p),
         connect(neuron.ground.g, neuron.n, channel.n)
     ] for channel in channels]

     input_connection = neuron.I.u ~ 0
    calcium_flux_connections = [[
            connect(channel.conductance.ca.p, neuron.ca.p),
            connect(neuron.ca.n, channel.conductance.ca.n),
     ] for channel in channels if hasproperty(channel.conductance, :ca) ]

     connections = vcat(channel_connections..., input_connection, calcium_flux_connections...)
     connected_system = compose(ODESystem(connections, t, name=nameof(neuron)), [channels..., neuron])
     return connected_system
end

function add_synapse_no_odesystem(channel, pre_neuron, post_neuron)
    pre_name = nameof(pre_neuron) 
    post_name = nameof(post_neuron)
    
    channel_connection = [
        connect(channel.pre, getproperty(pre_neuron, pre_name).p),
        connect(channel.post, getproperty(post_neuron, post_name).p),
    ]

    return [channel, channel_connection]
end

function add_synapse(channel, pre_neuron, post_neuron)
    pre_name = nameof(pre_neuron) 
    post_name = nameof(post_neuron)
    println("Names:  ", pre_name, "__", post_name)
    
    channel_connection = [
        connect(channel.pre, getproperty(pre_neuron, pre_name).p),
        connect(channel.post, getproperty(post_neuron, post_name).p),
    ]

    connected_system = compose(ODESystem(channel_connection, t, name=nameof(channel)),
        [channel, pre_neuron, post_neuron])
    return connected_system
end


function make_lif_synapse(pre_neuron, post_neuron, synapse; name)
    pre_name = nameof(pre_neuron) 
    post_name = nameof(post_neuron)
    println("Names:  ", pre_name, "__", post_name)
    
    eqs = [
        connect(synapse.pre, getproperty(pre_neuron, pre_name).p)
        connect(synapse.post, getproperty(post_neuron, post_name).p)
    ]

    connected_system =  compose(ODESystem(eqs, t; name=name), [pre_neuron, post_neuron, synapse])
    return connected_system
end

function build_LIF_network(connect1, connect2)
    network = []
    push!(network, connect1)
    push!(network, connect2)
    sys = compose(ODESystem([], t), [network...])
    return sys
end 

function validate_no_self_connections(connections::Dict)
    for (source, target) in keys(connections)
        if source == target
            error("Self-connection detected: neuron '$source' cannot connect to itself")
        end
    end
end

function validate_neuron_existence(connections::Dict, neurons)
    neuron_names = isa(neurons, Dict) ? keys(neurons) : ["n$i" for i in 1:length(neurons)]
    
    for (source, target) in keys(connections)
        source ∉ neuron_names && error("Neuron '$source' not found in neurons collection")
        target ∉ neuron_names && error("Neuron '$target' not found in neurons collection")
    end
end
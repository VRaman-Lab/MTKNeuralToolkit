
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

function build_neuron(neuron, input; channels)
     channel_connections = [[
         connect(channel.p, neuron.p),
         connect(neuron.ground.g, neuron.n, channel.n)
     ] for channel in channels]

     input_connection = connect(input.output, neuron.I)
     #calcium_connection = [[
     #                   connect(channel.reversal.ca.p, neuron.ca.p),            
     #                   connect(neuron.ca.n,  channel.reversal.ca.n),
     #                   #println("Triggered")            
     #] for channel in channels if hasproperty(channel.reversal, :ca) ]

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
     #calcium_connection = [[
     #                   connect(channel.reversal.ca.p, neuron.ca.p),            
     #                   connect(neuron.ca.n,  channel.reversal.ca.n)            
     #               ] for channel in channels if hasproperty(channel.reversal, :ca) ] 

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
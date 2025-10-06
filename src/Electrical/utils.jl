function build_channel(conductance, reversal;name)
    if conductance.p === nothing
        return build_channel_explicit(conductance;name, reversal=reversal)
    end
    @named p = Pin()
    @named n = Pin()
    connections = [
        connect(conductance.p, reversal.n),
        connect(conductance.n, n),
        connect(reversal.p, p)
    ]
    return compose(ODESystem(connections, t; name), [p,n,conductance,reversal])
end

function build_channel(conductance; name)
    if conductance.p === nothing
        return build_channel_explicit(conductance;name)
    end
    @named p = Pin()
    @named n = Pin()
    connections = [
        connect(conductance.p, p),
        connect(conductance.n, n)
    ]
    return compose(ODESystem(connections, t; name), [p, n, conductance])
end

function build_channel_explicit(conductance; name, reversal=nothing)
    @named p = Pin()
    @named n = Pin()
    connections = reversal === nothing ? [
        connect(conductance.oneport.p, p),
        connect(conductance.oneport.n, n)
    ] : [
        connect(conductance.oneport.p, reversal.n),
        connect(conductance.oneport.n, n),
        connect(reversal.p, p)
    ]
    return compose(ODESystem(connections, t; name=name), reversal === nothing ? [p, n, conductance] : [p, n, conductance, reversal])
end

function build_neuron(neuron, input; channels)
    channel_connections = [[
         connect(channel.p, neuron.oneport.p),
         connect(neuron.ground.g, neuron.oneport.n, channel.n)
     ] for channel in channels]
    input_connection = connect(input.output, neuron.I)

    calcium_flux_connections = [[
            connect(channel.conductance.ca.p, neuron.ca.p),
            connect(neuron.ca.n, channel.conductance.ca.n),          
     ] for channel in channels if hasproperty(channel.conductance, :ca) ]

     connections = vcat(channel_connections..., input_connection, calcium_flux_connections...)
     connected_system = System(connections, t, name=nameof(neuron); systems=[neuron, channels..., input])
     return connected_system
end

function build_neuron(neuron; channels)
    build_neuron(neuron, Constant(; name=:input, k=0.0); channels)
end

function add_synapse(channel, pre_neuron, post_neuron;)
    pre_name = nameof(pre_neuron) 
    post_name = nameof(post_neuron)
    
    channel_connection = [
        connect(channel.pre, getproperty(pre_neuron, pre_name).oneport.p), 
        connect(channel.post, getproperty(post_neuron, post_name).oneport.p),
    ]

    return channel_connection, channel
end
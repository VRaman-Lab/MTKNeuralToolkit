"""
    Build channel from gates: put in series with voltage.
"""
function build_channel_old(gate, Reversal)
    return @mtkmodel Channel begin
        @parameters begin
            g = g, [description = "Channel conductance"]
            E = E, [description = "Reversal Potential"]
        end
        @components begin
            p = Pin()
            n = Pin()
            reversal = Reversal(;E = E)
            conductance = gate(g=g, E=E)
        end
        @equations begin
            connect(conductance.p, reversal.n)
            connect(conductance.n, n)
            connect(reversal.p, p)
        end
    end
end


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

 # function build_neuron(neuron; channels, input)
 #     channel_connections = [[
 #         connect(channel.p, neuron.p),
 #         connect(neuron.ground.g, neuron.n, channel.n)
 #     ] for channel in channels]

 #     input_connection = connect(input.output, neuron.I)
 #     connections = vcat(channel_connections..., input_connection)
 #     connected_system = compose(ODESystem(connections, t, name=nameof(neuron)), [channels..., neuron,input])
 #     return connected_system
 # end

function build_neuron(neuron; channels, input)
     channel_connections = [[
         connect(channel.p, neuron.p),
         connect(neuron.ground.g, neuron.n, channel.n)
     ] for channel in channels]

     input_connection = connect(input.output, neuron.I)
     calcium_connection = [[
                        connect(channel.reversal.ca.p, neuron.ca.p),
                        connect(neuron.ca.n,  channel.reversal.ca.n)
                    ] for channel in channels if hasproperty(channel.reversal, :ca) ] # IF HAS CALCIUM NEED TO ADD TODO

    calcium_flux_connections = [[
            connect(channel.conductance.ca.p, neuron.ca.p),
            # connect(neuron.ca.n, channel.conductance.ca.p),
     ] for channel in channels if hasproperty(channel.conductance, :ca) ]

     connections = vcat(channel_connections..., input_connection, calcium_connection..., calcium_flux_connections...)
     connected_system = compose(ODESystem(connections, t, name=nameof(neuron)), [channels..., neuron,input])
     return connected_system
 end




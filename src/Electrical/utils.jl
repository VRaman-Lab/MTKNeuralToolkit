"
Builds a channel which conects a conductance and a reversal property
"
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

"
Subtype of build channel, doesn't take a reversal 
"

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

"
Builds a neuron which connects the channel to the nueron made in test files
"
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

"
Builds a neuron but makes a constant channel
"
function build_neuron(neuron; channels)
    build_neuron(neuron, Constant(; name=:input, k=0.0); channels)
end

"
Makes a synapse which connects a pre neuron to a post neuron 
"
function add_synapse(channel, pre_neuron, post_neuron;)
    pre_name = nameof(pre_neuron) 
    post_name = nameof(post_neuron)
    
    channel_connection = [
        connect(channel.pre, getproperty(pre_neuron, pre_name).oneport.p), 
        connect(channel.post, getproperty(post_neuron, post_name).oneport.p),
    ]

    return channel_connection, channel
end

"
Same as add_synapse but returns ODEProblem (works for LIF may be a issue to fix)
"
function make_lif_synapse(pre_neuron, post_neuron, synapse; name)
    pre_name = nameof(pre_neuron) 
    post_name = nameof(post_neuron)
    println("Names:  ", pre_name, "__", post_name)
    
    eqs = [
        connect(synapse.pre, getproperty(pre_neuron, pre_name).oneport.p)
        connect(synapse.post, getproperty(post_neuron, post_name).oneport.p)
    ]

    return compose(ODESystem(eqs, t; name), [pre_neuron, post_neuron, synapse])
end

function make_spike_callback(prob, neurons_or_idx)
    param_syms   = parameters(prob.f.sys)
    p_tunable, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)

    V_th_pidx    = findfirst(s -> contains(string(s), "V_th"),    param_syms)
    V_reset_pidx = findfirst(s -> contains(string(s), "V_reset"), param_syms)

    # resolve indices vs neuron systems
    v_indices = if eltype(neurons_or_idx) <: Integer
        neurons_or_idx
    else
        state_syms = unknowns(prob.f.sys)
        map(neurons_or_idx) do n
            name = string(nameof(n))
            sym  = state_syms[findfirst(s -> contains(string(s), name * "₊" * name * "₊oneport₊v"), state_syms)]
            variable_index(prob, sym)
        end
    end

    spike_times = [Float64[] for _ in v_indices]

    callbacks = map(enumerate(v_indices)) do (i, v_idx)
        ContinuousCallback(
            # read V_th live from integrator so remake'd params are respected
            (u, t, integrator) -> begin
                V_th = SciMLStructures.canonicalize(SciMLStructures.Tunable(), integrator.p)[1][V_th_pidx]
                u[v_idx] - V_th
            end,
            (integrator) -> begin
                p   = SciMLStructures.canonicalize(SciMLStructures.Tunable(), integrator.p)[1]
                V_reset = p[V_reset_pidx]
                integrator.u[v_idx] = V_reset
                push!(spike_times[i], integrator.t)
            end
        )
    end

    return CallbackSet(callbacks...), spike_times
end
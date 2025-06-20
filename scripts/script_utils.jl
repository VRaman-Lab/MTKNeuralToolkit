#include("script_types.jl")


function build_network(inpHH::Vector, inpLiu::Vector, inpIF::Vector, noinpHH::Int, noinpLiu::Int, noinpIF::Int,connections::Vector)
    iterator = 0
    synterator = 0
    neurons = Dict{String, Any}()
    network = []

    for inp in inpHH
        x = build_HH(inp; name=Symbol("n$iterator"))
        neurons["n$iterator"] = x
        #push!(network, x)
        iterator+=1
    end
    for inp in inpLiu
        x = build_Liu(inp; name=Symbol("n$iterator"))
        neurons["n$iterator"] = x
        #push!(network, x)
        iterator+=1
    end
    for inp in inpIF
        x = build_IF(inp; name=Symbol("n$iterator"))
        neurons["n$iterator"] = x
        #push!(network, x)
        iterator+=1
    end
    for h in 1:noinpHH
        x = build_HH(name=Symbol("n$iterator"))
        neurons["n$iterator"] = x
        #push!(network, x)
        iterator+=1
    end
    for h in 1:noinpLiu
        x = build_Liu(name=Symbol("n$iterator"))
        neurons["n$iterator"] = x
        #push!(network, x)
        iterator+=1
    end
    for h in 1:noinpIF
        x = build_IF(name=Symbol("n$iterator"))
        neurons["n$iterator"] = x
        #push!(network, x)
        iterator+=1
    end
    for (pre,post,type,weight) in connections
        #push!(network, put_synapse(neurons[pre], neurons[post], type, weight))
        x = put_synapse(neurons[pre], neurons[post], type, weight; name=Symbol("s$synterator"))
        synterator+=1
        push!(network, x)
    end
    final_system = compose(ODESystem([], t; name=:network), network)
    return structural_simplify(final_system)
end

function build_IF(input=nothing; name=:IF)
    string = ("Not implemented yet :(")
    println(string)
    println("Your code will crash in:")
    println("3")
    println("2")
    println("1")
    println("Never gonna give you up")
    return string
    #TODO Everything lol
end

function build_HH(input=nothing; name=:soma)

    Na = build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)      
    K = build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
    Leak = build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)

    fn=BasicSoma(; C=1, name = name)

    if input === nothing
        neur = build_neuron(fn; channels = [Na, K, Leak])
    else
        neur = build_neuron(fn, input; channels = [Na, K, Leak])
    end
    return(neur)
end

function build_Liu(input=nothing; name=:soma)

    Na =   build_channel(Liu.NaGates(;g=100, E = 50.0), FixedReversal(;E=50.0); name = :Na)
    KCa =  build_channel(Liu.KCaGates(;g=10.0, E = -80.0), FixedReversal(;E=-80.0); name = :KCa)
    CaS =  build_channel(Liu.CaSGates(;g=1.3), Liu.CalciumReversal(); name = :CaS)
    CaT =  build_channel(Liu.CaTGates(;g=3.0), Liu.CalciumReversal(); name = :CaT)
    K =    build_channel(Liu.KGates(;g=5.0, E = -80.0), FixedReversal(;E=-80.0); name = :K)
    DRK =  build_channel(Liu.DRKGates(;g=20.0, E = -80.0), FixedReversal(;E=-80.0); name = :KDR)
    H =    build_channel(Liu.HGates(;g=0.5, E = -20.0), FixedReversal(;E=-20.0); name = :H)
    Leak = build_channel(Liu.LeakGates(;g=0.1, E = -50.0), FixedReversal(;E=-50.0); name = :Leak)

    fn = Liu.CalciumSensitiveNeuron(; C=1, name = :soma)

    if input === nothing
        neur = build_neuron(fn;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    else
        neur = build_neuron(fn, input;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    end
    return(neur)
end

function put_synapse(pre, post, synapse_type::Symbol, weight::Float64; name=:syn, custom_synapse::Union{CustomSynapseParams, Nothing}=nothing)
    synapse_type in SYNAPSE_TYPES || throw(ArgumentError("Invalid synapse type"))
    if synapse_type == :Exc
        @named syn_channel = Synapse.E_syn_gate_preset(;g=weight, name =name)
    elseif synapse_type == :Inh
        @named syn_channel = Synapse.I_syn_gate_preset(;g=weight, name =name)
    elseif synapse_type == :Custom
        if custom_synapse === nothing
            throw(ArgumentError("If you want a custom synapse, you need to give a custom synapse, smartypants"))
        end
        @named syn_channel = custom_synapse(;g=weight, custom_synapse.E, custom_synapse.Vth, custom_synapse.k_, custom_synapse.sigma, name=name)
    end
    return add_synapse(syn_channel, pre, post)
end

function parse_sol_for_voltage(state_vars)
    neuron_to_states = Dict{String, Vector{Any}}()
    
    for state in state_vars
        state_str = string(state)
        
        if occursin(r"₊v\(t\)$", state_str)
            # Extract pattern: s0₊n0₊n0₊v(t)
            component_path = replace(state_str, r"₊v\(t\)$" => "")
            parts = split(component_path, "₊")
            
            if length(parts) >= 3
                # Get the final neuron identifier (n0, n1, n2, etc.)
                final_neuron = parts[end]
                
                if !haskey(neuron_to_states, final_neuron)
                    neuron_to_states[final_neuron] = Any[]
                end
                push!(neuron_to_states[final_neuron], state)
            end
        end
    end
    
    # Get first state for each neuron (sorted by string representation)
    first_states = Any[]
    for neuron_id in sort(collect(keys(neuron_to_states)))
        states_for_neuron = neuron_to_states[neuron_id]
        
        sorted_states = sort(states_for_neuron, by=string)
        push!(first_states, sorted_states[1])
    end
    
    return first_states
end

function parse_sol_for_membrane_voltages(sol::ODESolution)
    #ODESolution as input. Due to construction methodology, neurons might appear multiple times under different synapses if connected to differnet synapses.
    #These are just references to the same neuron though.
    #This function outputs an array consisting of all different neuron voltages in the system, only once for each.
    state_vars = unknowns(sol.prob.f.sys)
    voltage_states = parse_sol_for_voltage(state_vars)
    return voltage_states
end

function make_dense_layer(num_neurons::Int, neuron_type::Symbol, synapse_type::Symbol, weight = 0.1, pre_neuron=nothing, post_neuron=nothing,  custom_neuron::Union{CustomNeuronParams, Nothing}=nothing)
    neuron_type in NEURON_TYPES || throw(ArgumentError("Invalid neuron type"))
    synapse_type in SYNAPSE_TYPES || throw(ArgumentError("Invalid synapse type"))
    if post_neuron === pre_neuron === nothing
        throw(ArgumentError("One of pre_neuron or post_neuron must be provided. These can be a single neuron or an array of neurons."))
    end
    neurons = make_neurons_for_dense_layer(num_neurons, neuron_type)
    network = connect_neurons_for_dense_layer(pre_neuron, post_neuron, neurons, synapse_type, weight)
    return network
end

function make_neurons_for_dense_layer(num_neurons, neuron_type)
    neurons = []
    iterator = 0
    if neuron_type == :IF
        println("todo my bad lol")
    elseif neuron_type == :LIF
        println("todo my bad lol")
    elseif neuron_type == :HH
        for iterator in iterator:num_neurons
            push!(neurons, build_HH(;name=Symbol("d$iterator")))
        end
    elseif neuron_type == :Liu
        for iterator in iterator:num_neurons
            push!(neurons, build_Liu(;name=Symbol("d$iterator")))
        end    
    elseif neuron_type == :Custom
        if custom_neuron === nothing
            throw(ArgumentError("If you want a custom neuron, you need to give arguments for a custom neuron, smartypants"))
        end
        println("todo my bad lol")
    end
    return neurons
end

function connect_neurons_for_dense_layer(pre_neuron=nothing, post_neuron=nothing, neurons, synapse_type, weight)
    synterator = 0
    network = []
    if !isnothing(pre_neuron)
        for target in pre_neuron'

            
            for neuron in neurons
                push!(network, put_synapse(target, neuron, synapse_type, weight, name =Symbol("ds$synterator")))
                synterator+=1
            end
        end
    else
        for target in post_neuron
            for neuron in neurons
                push!(network, put_synapse(target, neuron, synapse_type, weight, name =Symbol("ds$synterator")))
                synterator+=1
            end
        end
    end
    return network
end

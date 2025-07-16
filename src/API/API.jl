module API

import ..MTKNeuralToolkit

import MTKNeuralToolkit.Config as config


function build_network(connections::Dict, neurons::Dict, check_connections=true)
    #To be used for custom neuron-synapse naming schemes
    if isempty(connections)
        error("Connections must be provided.")
    end
    if isempty(neurons)
        error("Neurons must be provided.")
    end
    validate_neuron_existence(connections, neurons)
    if check_connections validate_no_self_connections(connections) end
    network = create_network_from_connections(connections, neurons, [])
    final_system = compose(ODESystem([], t; name=:network), network)
    return structural_simplify(final_system)
end

"""
    build_network(connections::Dict, neurons::Vector, check_connections=true)

Build a network from a vector of neurons, automatically naming them n1, n2, etc.
Note: neuron names in connections must match the generated names.
"""

function build_network(connections::Dict, neurons::Vector, check_connections=true)
    #To be used for programmatic neuron naming
    #Beware that neurons will not have the same name when declaring connections as they will have symbolically within the system
    if isempty(connections)
        error("Connections must be provided.")
    end
    if isempty(neurons)
        error("Neurons must be provided.")
    end
    validate_neuron_existence(connections, neurons)
    if check_connections validate_no_self_connections(connections) end
    neurons_dict = Dict("n$i" => neuron for (i, neuron) in enumerate(neurons))
    network = []
    network = create_network_from_connections(connections, neurons, neurons_dict)
    final_system = compose(ODESystem([], t; name=:network), network)
    return structural_simplify(final_system)
end

"""
Internal: Wire synapses between neurons based on connection specifications.
"""

function create_network_from_connections(connections::Dict{Tuple{String, String}, @NamedTuple{type::Symbol, weight::Float64}}, neurons::Dict, network::Vector)
    for ((pre, post), (conn_params)) in connections
        x = put_synapse(neurons[pre], neurons[post], conn_params.type, conn_params.weight; name=Symbol("s_$(pre)$(post)"))
        push!(network, x)
    end
    for (_,neuron) in neurons
        push!(network, neuron)
    end
    return network
end

function create_network_from_connections(connections::Dict{Tuple{String, String}, Vector{@NamedTuple{type::Symbol, weight::Float64}}}, neurons::Dict, network::Vector)
    for ((pre, post), synapses) in connections
        for (i, synapse_params) in enumerate(synapses)
            x = put_synapse(neurons[pre], neurons[post], synapse_params.type, synapse_params.weight; 
                          name=Symbol("s_$(pre)$(post)_$(synapse_params.type)_$i"))
            push!(network, x)
        end
    end
    for (_,neuron) in neurons
        push!(network, neuron)
    end
    return network
end

function create_network_from_connections(connections::Dict{Tuple{String, String}, Function}, neurons::Dict, network::Vector)
    for ((pre, post), (synapse_func)) in connections
        @named x = synapse_func()
        y = add_synapse(x, neurons[pre], neurons[post])
        push!(network, y)
    end
    for (_,neuron) in neurons
        push!(network, neuron)
    end
    return network
end

"""
    put_synapse(pre, post, synapse_type::Symbol, weight::Float64; kwargs...)

Create and connect a synapse between two neurons.
Supports :Exc, :Inh, :Chol, :Glut, and :Custom synapse types.
"""

function put_synapse(pre, post, synapse_type::Symbol, weight::Float64; name=:syn, custom_synapse::Union{CustomSynapseParams, Nothing}=nothing)
    synapse_type in SYNAPSE_TYPES || throw(ArgumentError("Invalid synapse type"))
    if synapse_type == :Exc
        @named syn_channel = Synapse.E_syn_gate_preset(;g=weight, name =name)
    elseif synapse_type == :Inh
        @named syn_channel = Synapse.I_syn_gate_preset(;g=weight, name =name)
    elseif synapse_type == :Chol
        @named syn_channel = Synapse.CholinergicSynapse(;g=weight, name =name)
    elseif synapse_type == :Glut
        @named syn_channel = Synapse.GlutamatergicSynapse(;g=weight, name =name)
    elseif synapse_type == :Custom
        if custom_synapse === nothing
            throw(ArgumentError("If you want a custom synapse, you need to give a custom synapse, smartypants"))
        end
        @named syn_channel = custom_synapse(;g=weight, custom_synapse.E, custom_synapse.Vth, custom_synapse.k_, custom_synapse.sigma, name=name)
    end
    return add_synapse(syn_channel, pre, post)
end

"""
Placeholder for integrate-and-fire neuron implementation.
"""

function build_IF(input=nothing; name=:IF)
    string = ("Not implemented yet :(")
    println(string)
    println("Your code will crash in:")
    println("3")
    println("2")
    println("1")
    error("Never gonna give you up")
    #TODO Everything lol
end

"""
    build_HH(input=nothing; name=:soma, config=HHConfig())

Build a Hodgkin-Huxley neuron with Na, K, and leak channels.
Optional input stimulus and customizable parameters via config.
"""

function build_HH(input=nothing; name=:soma, config=config.HHConfig())

    Na = build_channel(HH.NaGates(;g=config.Na_g, E=config.Na_E), FixedReversal(;E=config.Na_E); name = :Na)      
    K = build_channel(HH.KGates(;g=config.K_g, E=config.K_E), FixedReversal(;E=config.K_E); name = :K)
    Leak = build_channel(HH.LGates(;g=config.Leak_g, E=config.Leak_E), FixedReversal(;E=config.Leak_E); name = :Leak)

    fn=BasicSoma(; C=1, name = name)

    if input === nothing
        neur = build_neuron(fn; channels = [Na, K, Leak])
    else
        neur = build_neuron(fn, input; channels = [Na, K, Leak])
    end
    return(neur)
end

"""
    build_Prinz(input=nothing; name=:soma, config=PrinzConfig())

Build a Prinz STG neuron model with calcium dynamics.
Commonly used for central pattern generator networks.
"""

function build_Liu(input=nothing; name=:soma, config=config.LiuConfig())

    Na =   build_channel(Liu.NaGates(;g=config.Na_g, E=config.Na_E), FixedReversal(;E=config.Na_E); name = :Na)
    KCa =  build_channel(Liu.KCaGates(;g=config.KCa_g, E=config.KCa_E), FixedReversal(;E=config.KCa_E); name = :KCa)
    CaS =  build_ca_channel(Liu.CaSGates(;g=config.CaS_g); name = :CaS)
    CaT =  build_ca_channel(Liu.CaTGates(;g=config.CaT_g); name = :CaT)
    K =    build_channel(Liu.KGates(;g=config.K_g, E=config.K_E), FixedReversal(;E=config.K_E); name = :K)
    DRK =  build_channel(Liu.DRKGates(;g=config.DRK_g, E=config.DRK_E), FixedReversal(;E=config.DRK_E); name = :KDR)
    H  = build_channel(Liu.HGates(;g=config.H_g, E=config.H_E), FixedReversal(;E=config.H_E); name = :H )
    Leak = build_channel(Liu.LeakGates(;g=config.Leak_g, E=config.Leak_E), FixedReversal(;E=config.Leak_E); name = :Leak)

    fn = Liu.CalciumSensitiveNeuron(; C=1, name = name)

    if input === nothing
        neur = build_neuron(fn;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    else
        neur = build_neuron(fn, input;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    end
    return(neur)
end

"""
Internal: Extract voltage states from system unknowns, handling duplicates.
"""

function build_Prinz(input=nothing; name=:soma, config=config.PrinzConfig())

    Na =   build_channel(Prinz.NaGates(;g=config.Na_g, E=config.Na_E), FixedReversal(;E=config.Na_E); name = :Na)
    KCa =  build_channel(Prinz.KCaGates(;g=config.KCa_g, E=config.KCa_E), FixedReversal(;E=config.KCa_E); name = :KCa)
    CaS =  build_ca_channel(Prinz.CaSGates(;g=config.CaS_g); name = :CaS)
    CaT =  build_ca_channel(Prinz.CaTGates(;g=config.CaT_g); name = :CaT)
    K =    build_channel(Prinz.KGates(;g=config.K_g, E=config.K_E), FixedReversal(;E=config.K_E); name = :K)
    DRK =  build_channel(Prinz.DRKGates(;g=config.DRK_g, E=config.DRK_E), FixedReversal(;E=config.DRK_E); name = :KDR)
    H  = build_channel(Prinz.HGates(;g=config.H_g, E=config.H_E), FixedReversal(;E=config.H_E); name = :H )
    Leak = build_channel(Prinz.LeakGates(;g=config.Leak_g, E=config.Leak_E), FixedReversal(;E=config.Leak_E); name = :Leak)

    fn = Prinz.CalciumSensitiveNeuron(; C=config.C, Ca=config.Ca0, V=config.V0, name = name)

    if input === nothing
        neur = build_neuron(fn;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    else
        neur = build_neuron(fn, input;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    end
    return(neur)
end

"""
    parse_sol_for_membrane_voltages(sol::ODESolution)

Extract unique membrane voltage traces from ODE solution.
Returns one voltage per neuron, avoiding duplicates from multiple connections.
"""

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

"""
    inspect_network(prob::Union{ODEProblem,ODESystem})

Display network structure including neurons and their synaptic connections.
Useful for debugging and verifying network topology.
"""

function inspect_network(prob::Union{ODEProblem,ODESystem})
    sys = if prob isa ODEProblem
        prob.f.sys
    else
        prob 
    end
    
    states = unknowns(sys)
    
    neurons = Dict{String, Vector{Any}}()
    for state in states
        state_str = string(state)
        if occursin(r"₊v\(t\)$", state_str)
            parts = split(replace(state_str, r"₊v\(t\)$" => ""), "₊")
            neuron_name = parts[end]
            
            if !haskey(neurons, neuron_name)
                neurons[neuron_name] = Any[]
            end
            push!(neurons[neuron_name], state)
        end
    end
    
    neuron_synapses = Dict{String, Vector{String}}()
    for state in states
        state_str = string(state)
        for (neuron_name, _) in neurons
            if occursin(neuron_name, state_str) && occursin(r"s\d+", state_str)
                match_obj = match(r"(s\d+)", state_str)
                if match_obj !== nothing
                    syn_name = match_obj[1]
                    if !haskey(neuron_synapses, neuron_name)
                        neuron_synapses[neuron_name] = String[]
                    end
                    if !(syn_name in neuron_synapses[neuron_name])
                        push!(neuron_synapses[neuron_name], syn_name)
                    end
                end
            end
        end
    end

    println("=== Network Inspection ===")
    println("\nNeurons ($(length(neurons))):")
    for (name, states) in sort(neurons)
        println("  $name: $(length(states)) state(s)")
        if haskey(neuron_synapses, name)
            println("    Connected synapses: ", join(sort(neuron_synapses[name]), ", "))
        end
    end
    println("__________________________")
    
    return (neurons=neurons, neuron_synapses=neuron_synapses, all_states=states)
end

end
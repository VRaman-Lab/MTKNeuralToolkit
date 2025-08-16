import ..Config as config
import ..HodgkinHuxley as HH_module
import ..Liu as Liu_module
import ..Prinz as Prinz_module
import ..Synapse as Synapse
import ..Types
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq
#

#=function build_network(connections::Dict{Tuple{Int,Int}, Vector{@NamedTuple{type::Symbol, weight::Float64}}}, neurons::Vector)
    synapses = mapreduce(vcat, connections) do ((pre_idx, post_idx), synapse_list)
        map(enumerate(synapse_list)) do (i, synapse_params)
            put_synapse(neurons[pre_idx], neurons[post_idx], synapse_params.type, synapse_params.weight; 
                       name=Symbol("s_$(pre_idx)$(post_idx)_$(synapse_params.type)_$i"))
        end
    end
    
    network_components = vcat(collect(values(neurons)), synapses)
    final_system = compose(ODESystem([], t; name=:network), network_components)
    return structural_simplify(final_system)
end=#

function build_network_split(connections::Dict{<:Tuple, Vector{@NamedTuple{type::Symbol, weight::Float64}}}, neurons::Union{Vector,Dict})
    # First, structural_simplify all neurons
    println("Simplifying neurons...")
    simplified_neurons = Dict{Any,Any}()
    for (key, neuron) in pairs(neurons)
        simplified_neurons[key] = structural_simplify(neuron)
    end
    
    # Create synapses using simplified neurons
    println("Creating synapses...")
    synapses = mapreduce(vcat, connections) do ((pre, post), synapse_list)
        map(enumerate(synapse_list)) do (i, synapse_params)
            put_synapse(simplified_neurons[pre], simplified_neurons[post], synapse_params.type, synapse_params.weight; 
                       name=Symbol("s_$(pre)$(post)_$(synapse_params.type)_$i"))
        end
    end
    
    # Simplify each synapse individually
    println("Simplifying synapses...")
    simplified_synapses = map(synapses) do synapse
        structural_simplify(synapse)
    end
    
    # Compose simplified components
    network_components = vcat(collect(values(simplified_neurons)), simplified_synapses)
    final_system = compose(ODESystem([], t; name=:network), network_components)
    
    # Final network-level simplification
    #println("Final network structural_simplify...")
    ss = structural_simplify(final_system)
    return ss
end

function build_network2(connections::Dict{<:Tuple, Vector{@NamedTuple{type::Symbol, weight::Float64}}}, neurons::Union{Vector,Dict})
    synapses = mapreduce(vcat, connections) do ((pre, post), synapse_list)
        map(enumerate(synapse_list)) do (i, synapse_params)
            put_synapse(neurons[pre], neurons[post], synapse_params.type, synapse_params.weight; 
                       name=Symbol("s$(pre)$(post)_$(synapse_params.type)$i"))
        end
    end
    
    network_components = vcat(collect(values(neurons)), synapses)
    #final_system = compose(synapses, t; name=:network)
    #final_system = System(Equation[], t; systems=synapses, name=:network)
    #inal_system = compose(System([], t; name=:network), synapses)
    #return structural_simplify(final_system)
    final_system = System(Equation[], t; systems=network_components, name=:network)
    return structural_simplify(final_system)
end


function build_network(connections::Dict, neurons)
    synapses = mapreduce(vcat, connections) do ((pre_idx, post_idx), synapse_funcs)
        map(enumerate(synapse_funcs)) do (i, synapse_func)
            syn_instance = synapse_func()
            add_synapse(syn_instance, neurons[pre_idx], neurons[post_idx])
        end
    end
    
    #network_components = vcat(neurons, synapses)
    final_system = compose(ODESystem([], t; name=:network), synapses)
    return structural_simplify(final_system)
end

function build_network(connections::Dict{<:Tuple, Vector{@NamedTuple{type::Symbol, weight::Float64}}}, neurons::Union{Vector,Dict})
    s_connections::Vector{Equation} = []
    synapses = []
    
    for ((pre, post), synapse_list) in connections
        for (i, synapse_params) in enumerate(synapse_list)
            data = put_synapse(neurons[pre], neurons[post], synapse_params.type, synapse_params.weight; 
                       name=Symbol("s$(pre)$(post)_$(synapse_params.type)$i"))

            append!(s_connections, data[1])
            push!(synapses, data[2])
        end
    end
    neuron_systems = neurons isa Dict ? collect(values(neurons)) : neurons
    network_system = System(Equation[], t; 
                           systems=[neuron_systems..., synapses...], 
                           name=:network)
    
    # Then add connections using extend
    final_system = extend(network_system, System(s_connections, t, name=:connections))
    
    return structural_simplify(final_system)
end

function put_synapse(pre, post, synapse_type::Symbol, weight::Float64; name=:Custom, E=nothing, Vth=nothing, k_=nothing,sigma=nothing)
    synapse_type in Types.SYNAPSE_TYPES || throw(ArgumentError("Invalid synapse type"))
    if synapse_type == :Exc
        @named syn_channel = Synapse.E_syn_gate_preset(;g=weight, name =name)
    elseif synapse_type == :Inh
        @named syn_channel = Synapse.I_syn_gate_preset(;g=weight, name =name)
    elseif synapse_type == :Chol
        @named syn_channel = Synapse.CholinergicSynapse(;g=weight, name =name)
    elseif synapse_type == :Glut
        @named syn_channel = Synapse.GlutamatergicSynapse(;g=weight, name =name)
    elseif synapse_type == :Custom
        @named syn_channel = custom_synapse(;g=weight, E, Vth, k_, sigma, name=name)
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

    Na = build_channel(HH_module.NaGates(;g=config.Na_g, E=config.Na_E), FixedReversal(;E=config.Na_E); name = :Na)      
    K = build_channel(HH_module.KGates(;g=config.K_g, E=config.K_E), FixedReversal(;E=config.K_E); name = :K)
    Leak = build_channel(HH_module.LGates(;g=config.Leak_g, E=config.Leak_E), FixedReversal(;E=config.Leak_E); name = :Leak)

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

    Na =   build_channel(Liu_module.NaGates(;g=config.Na_g, E=config.Na_E), FixedReversal(;E=config.Na_E); name = :Na)
    KCa =  build_channel(Liu_module.KCaGates(;g=config.KCa_g, E=config.KCa_E), FixedReversal(;E=config.KCa_E); name = :KCa)
    CaS =  build_channel(Liu_module.CaSChannel(;g=config.CaS_g); name = :CaS)
    CaT =  build_channel(Liu_module.CaTChannel(;g=config.CaT_g); name = :CaT)
    K =    build_channel(Liu_module.KGates(;g=config.K_g, E=config.K_E), FixedReversal(;E=config.K_E); name = :K)
    DRK =  build_channel(Liu_module.DRKGates(;g=config.DRK_g, E=config.DRK_E), FixedReversal(;E=config.DRK_E); name = :KDR)
    H  = build_channel(Liu_module.HGates(;g=config.H_g, E=config.H_E), FixedReversal(;E=config.H_E); name = :H )
    Leak = build_channel(Liu_module.LeakGates(;g=config.Leak_g, E=config.Leak_E), FixedReversal(;E=config.Leak_E); name = :Leak)

    fn = Liu_module.CalciumSensitiveNeuron(; C=1, name = name)

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

    Na =   build_channel(Prinz_module.NaGates(;g=config.Na_g, E=config.Na_E), FixedReversal(;E=config.Na_E); name = :Na)
    KCa =  build_channel(Prinz_module.KCaGates(;g=config.KCa_g, E=config.KCa_E), FixedReversal(;E=config.KCa_E); name = :KCa)
    CaS =  build_channel(Prinz_module.CaS(;g=config.CaS_g); name = :CaS)
    CaT =  build_channel(Prinz_module.CaT(;g=config.CaT_g); name = :CaT)
    K =    build_channel(Prinz_module.KGates(;g=config.K_g, E=config.K_E), FixedReversal(;E=config.K_E); name = :K)
    DRK =  build_channel(Prinz_module.DRKGates(;g=config.DRK_g, E=config.DRK_E), FixedReversal(;E=config.DRK_E); name = :KDR)
    H  = build_channel(Prinz_module.HGates(;g=config.H_g, E=config.H_E), FixedReversal(;E=config.H_E); name = :H )
    Leak = build_channel(Prinz_module.LeakGates(;g=config.Leak_g, E=config.Leak_E), FixedReversal(;E=config.Leak_E); name = :Leak)

    fn = Prinz_module.CalciumSensitiveNeuron(; C=config.C, Ca=config.Ca0, V=config.V0, name = name)

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
function select_time_varying_function(inp::String, default_const=1.0)
   functions = Dict(
       "sin" => sin,
       "cos" => cos,
       "exp" => exp,
       "log" => log,
       "constant" => t -> default_const
   )
   return functions[inp]
end

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

function build_IF(input=nothing, name=:IF)
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
        neur = build_neuron(fn, inp;  channels = [KCa, Na, CaS, CaT, K, DRK, H, Leak])
    end
    return(neur)
end

function put_synapse(pre, post, isE::Bool, weight::Float64; name=:syn)
    if isE
        @named syn_channel = Synapse.E_syn_gates(;g=weight, name =name)
    else
        @named syn_channel = Synapse.I_syn_gates(;g=weight, name =name)
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
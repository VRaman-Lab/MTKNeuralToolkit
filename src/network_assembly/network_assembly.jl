import ..Config as config
import ..HodgkinHuxley as HH_module
import ..Liu as Liu_module
import ..Prinz as Prinz_module
import ..IntegrateAndFire as IF_module
import ..Synapse as Synapse
import ..Types
import ..IntegrateAndFire as IaF
using ModelingToolkit
using ModelingToolkit: t_nounits as t
using OrdinaryDiffEq

function build_network(connections::Dict, neurons)
    s_connections::Vector{Equation} = []
    synapses = []
    
    for ((pre_idx, post_idx), synapse_funcs) in connections
        for (i, synapse_func) in enumerate(synapse_funcs)
            syn_instance = synapse_func()
            conn_eqs, syn = add_synapse(syn_instance, neurons[pre_idx], neurons[post_idx])
            
            append!(s_connections, conn_eqs)
            push!(synapses, syn)
        end
    end
    
    neuron_systems = neurons isa Dict ? collect(values(neurons)) : neurons
    network_system = System(Equation[], t; 
                           systems=[neuron_systems..., synapses...], 
                           name=:network)
    
    final_system = extend(network_system, System(s_connections, t, name=:connections))
    
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
    
    final_system = extend(network_system, System(s_connections, t, name=:connections))
    
    return structural_simplify(final_system)
end

function build_synapse(pre, post, synapse_type::Symbol, weight::Float64; name=:Custom, E=nothing, Vth=nothing, k_=nothing,sigma=nothing)
    synapse = put_synapse(pre, post, synapse_type, weight; name, E, Vth, k_, sigma)

    network_system = System(Equation[], t; 
                           systems=[pre, post, synapse], 
                           name=:network)
    
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
    elseif synapse_type == :LIF
        @named syn_channel = Synapse.LifSynapseComplex(;g_max=weight, name =name)
    end
    return add_synapse(syn_channel, pre, post)
end

function build_IF(input=nothing; name=:IF)
    IF = build_channel(IF_module.IF_channel(; E=0, name = :conductance), FixedReversal(; E=-65); name =:IF)
    fn = BasicSoma(; C=10, name = :soma)

    if input === nothing
        neur = build_neuron(fn; channels = [IF])
    else
        neur = build_neuron(fn, input; channels = [IF])
    end
    return(neur)
end

function build_LIF(input=nothing; name=:soma)
    LIF = build_channel(IF_module.IF_channel(; name = :conductance), FixedReversal(; E=-65); name =:LIF)
    fn = LIFSoma(; C=10, R = 1, name = name)

    if input === nothing
        neur = build_neuron(fn; channels = [LIF])
    else
        neur = build_neuron(fn, input; channels = [LIF])
    end
    return(neur)
end

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
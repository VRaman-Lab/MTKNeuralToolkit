const SYNAPSE_TYPES = (:Exc, :Inh, :Custom)
const NEURON_TYPES = (:IF, :LIF, :HH, :Liu, :Custom)

# Validation functions
is_valid_synapse(s) = s in SYNAPSE_TYPES
is_valid_neuron(n) = n in NEURON_TYPES

struct CustomSynapseParams
    E::Float64
    Vth::Float64
    k_::Float64
    sigma::Float64
    
    function CustomSynapseParams(E, Vth, k_, sigma)

        k_ > 0 || throw(ArgumentError("k_ must be positive"))
        sigma > 0 || throw(ArgumentError("sigma must be positive"))
        new(E, Vth, k_, sigma)
    end
end

struct CustomNeuronParams
    Channels::Vector
    Capacitance::Float64
    isCalcium::Bool
    
    function CustomNeuronParams(E, Vth, k_, sigma)

        !isempty(Channels) || throw(ArgumentError("Channels vector cannot be empty"))

        Capacitance > 0 || throw(ArgumentError("Capacitance must be positive"))    
        new(Channels, Capacitance, isCalcium)
    end
end

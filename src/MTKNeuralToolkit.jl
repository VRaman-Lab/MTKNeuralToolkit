module MTKNeuralToolkit
using ModelingToolkitNeuralNets
using Lux
using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random

include("Electrical/utils.jl")

export build_channel, build_channel_ann, build_RMM, build_neuron, build_calcium_neuron, build_minimal_channel, build_calcium_channel, build_full_channel, add_synapse, add_synapse_nu

include("Electrical/components.jl")

export NaGates, KGates, LGates, BasicSoma, FixedReversal, fixed_reversal

include("MixedIonic/components.jl")
export IonicPin, IonicPort, IonicTerminal, CalciumSensitiveNeuron

include("HodgkinHuxley/HodgkinHuxley.jl")
include("Liu/Liu.jl")

include("Synapse/Synapse.jl")

include("Types/Types.jl")

export SYNAPSE_TYPES, NEURON_TYPES, CustomSynapseParams

include("RMM/RMM.jl")

export RMMVec, RMMScal, RMMVecf

end

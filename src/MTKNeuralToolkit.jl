module MTKNeuralToolkit
using ModelingToolkitNeuralNets
using Lux
using ModelingToolkit
using SciCompDSL
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random

include("Electrical/utils.jl")

export build_channel, build_channel_explicit, build_neuron, add_synapse, make_lif_synapse, make_spike_callback, make_smooth_spike_callback
include("Electrical/components.jl")

export NaGates, KGates, LGates, BasicSoma, LIFSoma, reset_function, rrule, frule, FixedReversal, fixed_reversal

include("MixedIonic/components.jl")
export IonicPin, IonicPort, IonicTerminal, CalciumSensitiveNeuron, DirectionalTwoPort, BiDirectionalTwoPort

include("HodgkinHuxley/HodgkinHuxley.jl")
include("IntegrateAndFire/IntegrateAndFire.jl")
include("Liu/Liu.jl")

include("Synapse/Synapse.jl")

include("Types/Types.jl")

export SYNAPSE_TYPES

include("Loss/loss.jl")


include("GroundTruth/GroundTruth.jl")

export generate_groundtruth_system

include("RMM/RMM.jl")

export Full_RMM

include("Prinz/Prinz.jl")

include("Config/Config.jl")

include("network_assembly/network_assembly.jl")

export build_network, put_synapse, build_IF, build_LIF, build_HH, build_Liu, build_Prinz

end


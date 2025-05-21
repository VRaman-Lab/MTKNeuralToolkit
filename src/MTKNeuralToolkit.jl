module MTKNeuralToolkit

using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D

include("Electrical/utils.jl")

export build_channel, build_neuron, build_calcium_neuron, build_minimal_channel, build_calcium_channel, build_full_channel

include("Electrical/components.jl")

export NaGates, KGates, LGates, BasicSoma, FixedReversal, fixed_reversal

include("MixedIonic/components.jl")
export IonicPin, IonicPort, IonicTerminal, CalciumSensitiveNeuron

include("HodgkinHuxley/HodgkinHuxley.jl")
include("Liu/Liu.jl")
end

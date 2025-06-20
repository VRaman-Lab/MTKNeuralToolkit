module RMM


using ModelingToolkitNeuralNets
using Lux
using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D

include("LTI.jl")
include("ANN.jl")

end
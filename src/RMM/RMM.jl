module RMM


using ModelingToolkitNeuralNets
using Lux
using ModelingToolkit
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum, RealInputArray, RealOutputArray
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random

using OrdinaryDiffEq

include("channels.jl")
include("rmm_utils.jl")
end
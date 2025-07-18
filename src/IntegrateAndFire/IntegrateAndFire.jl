module IntegrateAndFire

import ..MTKNeuralToolkit
using Pkg 

using ModelingToolkit
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum, Step
using ModelingToolkit: t_nounits as t, D_nounits as D

include("channels.jl")

end
module Liu
import ..MTKNeuralToolkit: IonicPort, IonicPin, IonicGround, IonicTerminal


using ModelingToolkit
using SciCompDSL
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D




include("soma.jl")
include("channels.jl")

end

module  TestLoss
import ..MTKNeuralToolkit: IonicPort, IonicPin, IonicGround, IonicTerminal


using ModelingToolkit
using MTKNeuralToolkit
using Mooncake
using Zygote
using Plots
using ChainRulesCore
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using ForwardDiff
using DiffEqBase
using ReverseDiff
using OptimizationOptimJL
using OptimizationOptimisers
using SymbolicIndexingInterface
using SciMLStructures
using ComponentArrays
using SciMLSensitivity
using LinearAlgebra
using Optimization, OptimizationOptimisers, Optimisers
using OptimizationBBO
using Statistics: mean
using SciMLStructures: Tunable, canonicalize, replace, replace!
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D
using CUDA
using ProgressMeter

include("LossUtils.jl")
include("MultiParamFinite.jl")
include("FiniteDiff_test.jl")
include("ForwardDiff_test.jl")
include("Zygote_test.jl")
include("MultiParamBBO.jl")
include("CudaZygote.jl")
include("MulitParamForward.jl")

end

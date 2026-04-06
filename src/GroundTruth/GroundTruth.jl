module GroundTruth
import ..MTKNeuralToolkit: IonicPort, IonicPin, IonicGround, IonicTerminal

using ModelingToolkit
using MTKNeuralToolkit
using Zygote
using Plots
using ChainRulesCore
using OrdinaryDiffEq
using OrdinaryDiffEqNonlinearSolve
using Statistics 
using Optimization
using OptimizationOptimJL
using OptimizationOptimisers
using SymbolicIndexingInterface
using SciMLStructures
using ComponentArrays
using SciMLSensitivity
using SciMLStructures: Tunable, canonicalize, replace, replace!
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum
using ModelingToolkit: t_nounits as t, D_nounits as D

include("GenerateGroundTruth.jl")

end
module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput, Constant, RealOutput, RealInputArray
import ModelingToolkitStandardLibrary.Electrical: OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, SymbolicT, ImperativeAffect
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
using DynamicQuantities
using DataFrames
import SymbolicUtils: scalarize
import Symbolics: Sym, Num

# ==========================================
# 1. Core Framework
# ==========================================
include("topology.jl")
export Scalar, Vectorized

include("components/electrical.jl")

include("components/channels.jl")
export Ground, Capacitor, CurrentSource, GenericChannel, GateSpec

include("components/calcium.jl")
include("components/synapses.jl")
include("network.jl")

export build_compartment, build_acausal_network, build_synapse_block

export Compartment, Network, SynapseSpec, CouplingSpec
export CaVChannel, KCaChannel, CalciumPool, CalciumTracker, NoCalcium
export ExpSynapse, VectorizedExpSynapse

# ==========================================
# 2. Standard Model Library (Submodules)
# ==========================================
include("library/HodgkinHuxley.jl")
export HodgkinHuxley

end

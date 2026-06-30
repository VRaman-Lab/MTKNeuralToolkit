module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput, Constant, RealOutput, RealInputArray, RealOutputArray
import ModelingToolkitStandardLibrary.Electrical: OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, SymbolicT, ImperativeAffect
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
using DynamicQuantities
using DataFrames
import SymbolicUtils: scalarize
import Symbolics: Sym, Num

include("BasicComponents.jl")
export Ground, OnePort, Pin, Capacitor, SpikingCapacitor, CurrentSource, FixedReversal 
export ChemicalSynapse, GapJunction, AlphaSynapse, SynapseSpec

export VectorizedPin, VectorizedOnePort

include("connections.jl")
export build_compartment, Compartment
export build_synapse
export build_acausal_network, build_synapse_block, CouplingSpec
export Vectorized, Scalar


include("tempgates.jl")
export GateSpec, GenericChannel

export ExpSynapse, VectorizedExpSynapse

end

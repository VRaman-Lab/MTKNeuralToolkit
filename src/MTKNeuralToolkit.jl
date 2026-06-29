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
export GenericChannel

include("connections.jl")
export build_compartment, Cell, Compartment, build_cell, build_network
export build_synapse
export build_acausal_network 

include("tempgates.jl")
export GateSpec, GenericChannel

export ExpSynapse

include("vectorization.jl")
export vectorize_system

include("loss_functions.jl")
export build_loss

end

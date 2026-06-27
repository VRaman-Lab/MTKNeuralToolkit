module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput, Constant, RealOutput, RealInputArray, RealOutputArray
import ModelingToolkitStandardLibrary.Electrical: Ground, OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, SymbolicT,ImperativeAffect
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
using DynamicQuantities
using DataFrames
import SymbolicUtils: scalarize
import Symbolics: Sym, Num


include("BasicComponents.jl")
export Ground, OnePort, Pin, Capacitor, SpikingCapacitor, CurrentSource, FixedReversal 
export ChemicalSynapse, GapJunction, VectorizedAlphaSynapse, AlphaSynapse

include("connections.jl")
export build_channel, build_compartment, build_floating_compartment, Cell, Compartment, build_cell, build_network



export build_synapse
export build_electrical_network, build_vectorized_network, build_fully_vectorized_network
# include("causal_connections.jl")
# export CausalSynapseGate, build_causal_synapse, VectorSynapsePopulation





include("tempgates.jl")
export GateSpec, GenericChannel
export nagates,lgates,kgates
export InlinedHHNeuron, VectorizedHHNeuron

include("loss_functions.jl")
export build_loss

end


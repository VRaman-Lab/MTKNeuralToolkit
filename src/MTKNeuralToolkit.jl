module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput, Constant, RealOutput, RealInputArray, RealOutputArray
import ModelingToolkitStandardLibrary.Electrical: Ground, OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, SymbolicT,ImperativeAffect
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
using DynamicQuantities
import SymbolicUtils: scalarize
import Symbolics: Sym, Num


include("BasicComponents.jl")
export Ground, OnePort, Pin, Capacitor, LIFCapacitor, CurrentSource, FixedReversal 
export AlphaSynapse, AbstractSynapseSpec, AlphaSynapseSpec, ChemicalSynapse, GapJunction, VectorizedAlphaSynapse

include("connections.jl")
export build_channel, build_neuron, connect_synapse, build_compartment



export build_synapse, EventSynapseGate
export neuron_connect, build_electrical_network, build_factored_synapse_network, build_vectorized_network
# include("causal_connections.jl")
# export CausalSynapseGate, build_causal_synapse, VectorSynapsePopulation





include("tempgates.jl")



# include("MixedIonic/components.jl")
# export IonicPin, IonicPort, IonicTerminal, CalciumSensitiveNeuron, DirectionalTwoPort, BiDirectionalTwoPort

# include("HodgkinHuxley/HodgkinHuxley.jl")
# include("IntegrateAndFire/IntegrateAndFire.jl")
# include("Liu/Liu.jl")

# include("Synapse/Synapse.jl")

# include("Types/Types.jl")

# export SYNAPSE_TYPES

# include("RMM/RMM.jl")

# export Full_RMM

# include("Prinz/Prinz.jl")

# include("Config/Config.jl")

# include("network_assembly/network_assembly.jl")

# export build_network, put_synapse, build_IF, build_HH, build_Liu, build_Prinz
#export PrinzConfig
export Discrete, Continuous
end


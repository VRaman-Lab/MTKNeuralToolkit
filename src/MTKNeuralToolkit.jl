module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput
import ModelingToolkitStandardLibrary.Electrical: Ground, OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect 
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
using DynamicQuantities

# Rules:
# EVERY oneport element should expose el.p and el.n pins.



include("BasicComponents.jl")
export Ground, OnePort, Pin, Capacitor, LIFCapacitor, CurrentSource, FixedReversal 

# include("Electrical/utils.jl")
include("connections.jl")
export build_channel, build_neuron, connect_synapse, build_compartment
export build_synapse, EventSynapseGate

# include("BasicComponents.jl")



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

end


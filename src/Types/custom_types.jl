const SYNAPSE_TYPES = (:Exc, :Inh, :Custom, :Chol, :Glut, :LIF)
const NEURON_TYPES = (:IF, :LIF, :HH, :Liu, :Custom)

# Validation functions
is_valid_synapse(s) = s in SYNAPSE_TYPES
is_valid_neuron(n) = n in NEURON_TYPES


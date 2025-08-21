const SYNAPSE_TYPES = (:Exc, :Inh, :Custom, :Chol, :Glut, :LIF)

# Validation functions
is_valid_synapse(s) = s in SYNAPSE_TYPES
is_valid_neuron(n) = n in NEURON_TYPES


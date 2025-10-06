const SYNAPSE_TYPES = (:Exc, :Inh, :Custom, :Chol, :Glut, :LIF)

is_valid_synapse(s) = s in SYNAPSE_TYPES


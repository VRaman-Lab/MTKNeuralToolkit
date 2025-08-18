const SYNAPSE_TYPES = (:Exc, :Inh, :Custom, :Chol, :Glut)

is_valid_synapse(s) = s in SYNAPSE_TYPES


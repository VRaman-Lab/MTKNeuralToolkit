struct Scalar end
struct Vectorized
    N::Int
end

# Topology helper functions
get_N(::Scalar) = nothing
get_N(v::Vectorized) = v.N

init_voltage(::Scalar, V_init) = V_init
init_voltage(v::Vectorized, V_init) = fill(V_init, v.N)

function create_pins(::Scalar)
    @named p = Pin(); @named n = Pin()
    return (p, n)
end
function create_pins(v::Vectorized)
    @named p = VectorizedPin(N=v.N); @named n = VectorizedPin(N=v.N)
    return (p, n)
end

function create_injectors(::Scalar)
    @named injector = CurrentSource(); @named syn_injector = CurrentSource()
    return (injector, syn_injector)
end
function create_injectors(v::Vectorized)
    @named injector = CurrentSource(topology=v)
    @named syn_injector = CurrentSource(topology=v)
    return (injector, syn_injector)
end

# Network grounding helpers
create_ground(::Scalar, name) = Ground(name=name)
create_ground(v::Vectorized, name) = Ground(topology=v, name=name)

ground_current(::Scalar) = 0.0
ground_current(v::Vectorized) = zeros(Float64, v.N)

broadcast_stim(::Scalar, stim) = stim
broadcast_stim(v::Vectorized, stim) = fill(stim, v.N)

# Synapse grounding helpers
function ground_undriven_syn!(eqs, ::Scalar, I_syn, driven_syn_targets)
    if !(I_syn in driven_syn_targets)
        push!(eqs, I_syn ~ 0.0)
    end
end
function ground_undriven_syn!(eqs, v::Vectorized, I_syn, driven_syn_targets)
    for j in 1:v.N
        i_syn_j = I_syn[j]
        if !(i_syn_j in driven_syn_targets)
            push!(eqs, i_syn_j ~ 0.0)
        end
    end
end

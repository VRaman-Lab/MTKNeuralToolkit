using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq

function VectorizedSynapseBug(; name, N::Int)
    @variables v(t) I(t) S(t)[1:N, 1:N]
    @parameters W[1:N, 1:N] = ones(N, N)

    # Add an algebraic variable I to trigger setsym during init
    eqs = [D(v) ~ -v, D(S) ~ -S, I ~ sum(S)]

    # Event updating an array index using Pre()
    affect = [S[1, 1] ~ Pre(S[1, 1]) + W[1, 1]]
    events = [[v ~ 0.0] => affect]

    return System(eqs, t; continuous_events=events, name=name)
end

@named sys = VectorizedSynapseBug(N=2)
sys_compiled = mtkcompile(sys)

prob = ODEProblem(sys_compiled, [], (0.0, 1.0))

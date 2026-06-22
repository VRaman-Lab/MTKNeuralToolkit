using MTKNeuralToolkit
import ModelingToolkitStandardLibrary.Blocks
using ModelingToolkit: mtkcompile, @named, System, Equation
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq


function build_hh_neurons(N)
    neurons = System[]
    for i in 1:N
        @named soma = Capacitor(C = 1.0)
        channels = System[]
        push!(channels, build_channel(nagates(name=:gate), FixedReversal(E = 50.0, name=:batt); name=Symbol(:na_, i)))
        push!(channels, build_channel(kgates(name=:gate),  FixedReversal(E = -77.0, name=:batt); name=Symbol(:k_, i)))
        push!(channels, build_channel(lgates(name=:gate),  FixedReversal(E = -54.4, name=:batt); name=Symbol(:l_, i)))
        nrn = build_compartment(soma, channels; name = Symbol(:nrn_, i))
        push!(neurons, nrn)
    end
    return neurons
end


N = 10
neurons = build_hh_neurons(N)


tau_mat = fill(5.0, N, N)
gmax_mat = fill(0.5 / N, N, N)


println("Compiling Full Network...")
W = fill(0.1, N, N); [W[i,i] = 0.0 for i in 1:N]
@named exc_syn_conn = VectorizedAlphaSynapse(N=N, W=W, tau=tau_mat, g_max=gmax_mat)
@named stim = Blocks.Sine(frequency = 0.05, amplitude = 15.0)
drivers = [(1, stim)]
net3 = build_vectorized_network(neurons, [exc_syn_conn]; drivers=drivers)
t3 = @elapsed @named net3_compiled = mtkcompile(net3)
prob3 = ODEProblem(net3_compiled, [], (0.0, 50.0))
println("Solving Full Network...")
@time sol3 = solve(prob3, Tsit5(); reltol=1e-3, abstol=1e-3);
println(" compile time was", t3)

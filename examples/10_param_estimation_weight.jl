# ==============================================================================
# Example 10: Parameter Estimation of a Synaptic Weight
# ==============================================================================

using MTKNeuralToolkit
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named
using ModelingToolkitStandardLibrary.Blocks: Sine
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using SciMLStructures: Tunable, canonicalize, replace
using SymbolicIndexingInterface: parameter_values, setp
using PreallocationTools
using Plots

# ------------------------------------------------------------------------------
# 1. Build the Network & Generate Target Data
# ------------------------------------------------------------------------------
top = Scalar()

function build_neuron(name::Symbol)
    @named cap  = Capacitor(topology=top, C=1.0)
    @named na   = SodiumChannel(topology=top)
    @named k    = PotassiumChannel(topology=top)
    @named leak = LeakChannel(topology=top)
    return build_compartment(cap, [na, k, leak]; name=name, V_init=-65.0, topology=top)
end

pre_neuron  = build_neuron(:pre_neuron)
post_neuron = build_neuron(:post_neuron)

true_g_max = 3.0
@named true_synapse = ExpSynapse(g_max=true_g_max, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

true_synapse_specs = [
    SynapseSpec(pre_neuron.interfaces.V, post_neuron.interfaces.V, 
                post_neuron.interfaces.I_syn, true_synapse)
]

@named driver = Sine(amplitude=8.0, frequency=0.05, offset=8.0)
drivers = [(1, driver)]

true_net = build_acausal_network([pre_neuron, post_neuron]; 
                            synapse_specs=true_synapse_specs, 
                            drivers=drivers, name=:true_net)

sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(sys, [], (0.0, 200.0))

timesteps = 0.0:0.5:200.0
sol = solve(true_prob, Tsit5(); saveat=timesteps)
target_data = Array(sol)

# ------------------------------------------------------------------------------
# 2. Setup the Optimization Problem
# ------------------------------------------------------------------------------
guess_g_max = 1.0
@named fit_synapse = ExpSynapse(g_max=guess_g_max, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

fit_synapse_specs = [
    SynapseSpec(pre_neuron.interfaces.V, post_neuron.interfaces.V, 
                post_neuron.interfaces.I_syn, fit_synapse)
]

fit_net = build_acausal_network([pre_neuron, post_neuron]; 
                            synapse_specs=fit_synapse_specs, 
                            drivers=drivers, name=:fit_net)

fit_sys = mtkcompile(fit_net.sys)
fit_prob = ODEProblem(fit_sys, [], (0.0, 200.0))

g_max_sym = fit_sys.fit_synapse.g_max
setter = setp(fit_prob, [g_max_sym])
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))

function loss(x, p)
    prob, timesteps, data, setter, diffcache = p
    ps = parameter_values(prob)
    buffer = get_tmp(diffcache, x)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps = replace(Tunable(), ps, buffer)
    setter(ps, x)
    newprob = remake(prob; p = ps)
    sol = solve(newprob, Tsit5(); saveat=timesteps)
    if size(Array(sol)) != size(data)
        return Inf
    end
    return sum(abs2, Array(sol) .- data) / size(data, 2)
end

opt_params = (fit_prob, timesteps, target_data, setter, diffcache)
adtype = AutoForwardDiff()
optfn = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optfn, [guess_g_max], opt_params)

# ------------------------------------------------------------------------------
# 3. Optimize and Plot
# ------------------------------------------------------------------------------
println("Solving with initial guess for visualization...")
# Capture the behavior before optimization to show how far it came
init_sol = solve(fit_prob, Tsit5(); saveat=timesteps)

println("Starting optimization to find synaptic weight...")
res = solve(optprob, BFGS(); maxiters=200)

println("True g_max: $true_g_max")
println("Recovered g_max: $(res.u[1])")

# Re-solve with optimized parameters
opt_ps = parameter_values(fit_prob)
opt_buffer = copy(canonicalize(Tunable(), opt_ps)[1])
opt_ps = replace(Tunable(), opt_ps, opt_buffer)
setter(opt_ps, res.u)
opt_prob_final = remake(fit_prob; p=opt_ps)

opt_sol = solve(opt_prob_final, Tsit5(); saveat=timesteps)

# In our 2-neuron HH network, the post-synaptic voltage is the 5th state vector element
p1 = plot(timesteps, target_data[5, :], label="True Post-synaptic V", lw=2, color=:black)
plot!(p1, timesteps, init_sol[5, :], label="Initial Guess", ls=:dot, lw=2, color=:gray)
plot!(p1, timesteps, opt_sol[5, :], label="Fitted Post-synaptic V", ls=:dash, lw=2, color=:red)
title!(p1, "Parameter Estimation of Synaptic Weight")
xlabel!("Time (ms)")
ylabel!("V (mV)")
display(p1)

# # **14.** Parameter Estimation: Heterogeneous Vectorized Population
#
# ## Introduction
# Here we fit the conductances of a larger, heterogeneous E/I microcircuit.
# We have 10 Excitatory and 3 Inhibitory neurons. To make the dynamics asynchronous, we use random synaptic weight matrices.
# We attempt a large-scale optimization: fitting the **heterogeneous** `gNa` arrays (13 parameters) 
# alongside the **shared** `gK` and `gleak` scalars (4 parameters) for a total of 17 parameters.

using MTKNeuralToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, Pre
using SymbolicIndexingInterface: getu
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using MTKNeuralToolkit: PEMObservationChannel
using ModelingToolkit: mtkcompile, @named, @component
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using SciMLStructures: Tunable, canonicalize, replace
using SymbolicIndexingInterface: parameter_values, setp
using PreallocationTools
using DataInterpolations
using SciMLBase
using Plots
using Markdown
using Random

# ## 1. Build the True System & Generate Data

N_E = 10
N_I = 3  
top_E = Vectorized(N_E)
top_I = Vectorized(N_I);

# Heterogeneous true sodium conductances
gNa_E_true = collect(range(110.0, 130.0, length=N_E))
gNa_I_true = collect(range(115.0, 125.0, length=N_I))

true_gK = 36.0
true_gleak = 0.3

function build_population(name::Symbol, top; gNa, gK, gleak, pem=false, itps=nothing, K=1.0)
    @named cap  = Capacitor(topology=top, C=1.0)
    @named na   = SodiumChannel(topology=top, g=gNa)
    @named k    = PotassiumChannel(topology=top, g=gK)
    @named leak = LeakChannel(topology=top, g=gleak)
    channels = [na, k, leak]
    if pem
        @named pem_ch = PEMObservationChannel(itps=itps, K_init=K, topology=top)
        push!(channels, pem_ch)
    end
    return build_compartment(cap, channels; name=name, V_init=-65.0, topology=top)
end

pop_E_true = build_population(:pop_E_true, top_E; gNa=gNa_E_true, gK=true_gK, gleak=true_gleak)
pop_I_true = build_population(:pop_I_true, top_I; gNa=gNa_I_true, gK=true_gK, gleak=true_gleak)

Random.seed!(42) 
W_EE = 0.5 .* rand(N_E, N_E)
W_EI = 1.0 .* rand(N_I, N_E)
W_IE = 2.0 .* rand(N_E, N_I)
W_II = 1.0 .* rand(N_I, N_I)

syn_EE = build_synapse_block(pop_E_true, pop_E_true, W_EE; name=:syn_EE, E_rev=0.0)
syn_EI = build_synapse_block(pop_E_true, pop_I_true, W_EI; name=:syn_EI, E_rev=0.0)
syn_IE = build_synapse_block(pop_I_true, pop_E_true, W_IE; name=:syn_IE, E_rev=-80.0)
syn_II = build_synapse_block(pop_I_true, pop_I_true, W_II; name=:syn_II, E_rev=-80.0)

synapse_specs = [syn_EE, syn_EI, syn_IE, syn_II]
drivers = [(1, 15.0)] # Strong kick to E population

true_net = build_acausal_network([pop_E_true, pop_I_true]; synapse_specs=synapse_specs, drivers=drivers, name=:true_net)
true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, 50.0), jac=true, sparse=true)
timesteps = 0.0:0.1:50.0

true_sol = solve(true_prob, Rosenbrock23(); saveat=timesteps)

V_data_E_mat = reduce(hcat, true_sol[true_sys.pop_E_true.cap.v])
V_data_I_mat = reduce(hcat, true_sol[true_sys.pop_I_true.cap.v])

itps_E = [LinearInterpolation(V_data_E_mat[i, :], timesteps) for i in 1:N_E]
itps_I = [LinearInterpolation(V_data_I_mat[i, :], timesteps) for i in 1:N_I];

# ## 2. Setup the PEM Optimization Problem
# We fit 17 parameters total: heterogeneous `gNa` arrays (length 10 and 3) and shared `gK`/`gleak` scalars.
# We intentionally guess terrible values (`gNa=10.0`, `gK=100.0`, `gleak=10.0`) that abolish spiking entirely.

guess_gNa_E = fill(10.0, N_E)
guess_gNa_I = fill(10.0, N_I)
guess_gK = 100.0
guess_gleak = 10.0

pop_E_fit = build_population(:pop_E_fit, top_E; gNa=guess_gNa_E, gK=guess_gK, gleak=guess_gleak, pem=true, itps=itps_E, K=2.0)
pop_I_fit = build_population(:pop_I_fit, top_I; gNa=guess_gNa_I, gK=guess_gK, gleak=guess_gleak, pem=true, itps=itps_I, K=2.0)

syn_EE_fit = build_synapse_block(pop_E_fit, pop_E_fit, W_EE; name=:syn_EE_fit, E_rev=0.0)
syn_EI_fit = build_synapse_block(pop_E_fit, pop_I_fit, W_EI; name=:syn_EI_fit, E_rev=0.0)
syn_IE_fit = build_synapse_block(pop_I_fit, pop_E_fit, W_IE; name=:syn_IE_fit, E_rev=-80.0)
syn_II_fit = build_synapse_block(pop_I_fit, pop_I_fit, W_II; name=:syn_II_fit, E_rev=-80.0)

synapse_specs_fit = [syn_EE_fit, syn_EI_fit, syn_IE_fit, syn_II_fit]

fit_net = build_acausal_network([pop_E_fit, pop_I_fit]; synapse_specs=synapse_specs_fit, drivers=drivers, name=:fit_net)
fit_sys = mtkcompile(fit_net.sys)
fit_prob = ODEProblem(fit_sys, [], (0.0, 50.0), jac=true, sparse=true);

# Extract symbols for the 17 parameters
gNa_E_sym = fit_sys.pop_E_fit.na.g
gK_E_sym  = fit_sys.pop_E_fit.k.g
gleak_E_sym = fit_sys.pop_E_fit.leak.g
gNa_I_sym = fit_sys.pop_I_fit.na.g
gK_I_sym  = fit_sys.pop_I_fit.k.g
gleak_I_sym = fit_sys.pop_I_fit.leak.g

setter = setp(fit_prob, [gNa_E_sym, gK_E_sym, gleak_E_sym, gNa_I_sym, gK_I_sym, gleak_I_sym])
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))

v_getter_E = getu(fit_prob, fit_sys.pop_E_fit.cap.v)
v_getter_I = getu(fit_prob, fit_sys.pop_I_fit.cap.v)

# ## 3. Define Loss Function & Optimize

function loss(x, p)
    prob, timesteps, V_data_E, V_data_I, setter, diffcache, v_getter_E, v_getter_I = p
    ps = parameter_values(prob)
    buffer = get_tmp(diffcache, x)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps = replace(Tunable(), ps, buffer)
    
    #Split the flat x vector into the 6 grouped chunks expected by the setter
    vals = [
        x[1:N_E],                      #gNa_E (array of 10)
        x[N_E + 1],                    #gK_E (scalar)
        x[N_E + 2],                    #gleak_E (scalar)
        x[N_E + 3 : N_E + N_I + 2],    #gNa_I (array of 3)
        x[N_E + N_I + 3],              #gK_I (scalar)
        x[N_E + N_I + 4]               #gleak_I (scalar)
    ]
    
    setter(ps, vals)
    newprob = remake(prob; p=ps)
    sol = solve(newprob, Rosenbrock23(); saveat=timesteps)
    if !SciMLBase.successful_retcode(sol.retcode)
        return Inf
    end
    V_fit_E = reduce(hcat, v_getter_E(sol))
    V_fit_I = reduce(hcat, v_getter_I(sol))
    return (sum(abs2, V_fit_E .- V_data_E) + sum(abs2, V_fit_I .- V_data_I)) / (size(V_data_E, 2) * (N_E + N_I))
end


# Flatten the 17 initial guesses into a single vector for the optimizer
x0 = vcat(guess_gNa_E, [guess_gK, guess_gleak], guess_gNa_I, [guess_gK, guess_gleak])

opt_params = (fit_prob, timesteps, V_data_E_mat, V_data_I_mat, setter, diffcache, v_getter_E, v_getter_I)

adtype = AutoForwardDiff()
optfn = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optfn, x0, opt_params)

# ## 4. Optimize and Plot

println("Starting optimization (17 parameters)...")
res = solve(optprob, BFGS(); maxiters=100)

# Extract recovered parameters
opt_gNa_E = res.u[1:N_E]
opt_gK_E  = res.u[N_E + 1]
opt_gleak_E = res.u[N_E + 2]
opt_gNa_I = res.u[N_E + 3 : N_E + N_I + 2]
opt_gK_I  = res.u[N_E + N_I + 3]
opt_gleak_I = res.u[N_E + N_I + 4]

# --- Pure simulation for INITIAL GUESS (No PEM) ---
pop_E_init = build_population(:pop_E_init, top_E; gNa=guess_gNa_E, gK=guess_gK, gleak=guess_gleak)
pop_I_init = build_population(:pop_I_init, top_I; gNa=guess_gNa_I, gK=guess_gK, gleak=guess_gleak)

syn_EE_init = build_synapse_block(pop_E_init, pop_E_init, W_EE; name=:syn_EE_init, E_rev=0.0)
syn_EI_init = build_synapse_block(pop_E_init, pop_I_init, W_EI; name=:syn_EI_init, E_rev=0.0)
syn_IE_init = build_synapse_block(pop_I_init, pop_E_init, W_IE; name=:syn_IE_init, E_rev=-80.0)
syn_II_init = build_synapse_block(pop_I_init, pop_I_init, W_II; name=:syn_II_init, E_rev=-80.0)
synapse_specs_init = [syn_EE_init, syn_EI_init, syn_IE_init, syn_II_init]

init_net = build_acausal_network([pop_E_init, pop_I_init]; synapse_specs=synapse_specs_init, drivers=drivers, name=:init_net)
init_sys = mtkcompile(init_net.sys)
init_prob = ODEProblem(init_sys, [], (0.0, 50.0), jac=true, sparse=true)
init_sol = solve(init_prob, Rosenbrock23(); saveat=timesteps)

# --- Pure simulation for RECOVERED PARAMETERS (No PEM) ---
pop_E_eval = build_population(:pop_E_eval, top_E; gNa=opt_gNa_E, gK=opt_gK_E, gleak=opt_gleak_E)
pop_I_eval = build_population(:pop_I_eval, top_I; gNa=opt_gNa_I, gK=opt_gK_I, gleak=opt_gleak_I)

syn_EE_eval = build_synapse_block(pop_E_eval, pop_E_eval, W_EE; name=:syn_EE_eval, E_rev=0.0)
syn_EI_eval = build_synapse_block(pop_E_eval, pop_I_eval, W_EI; name=:syn_EI_eval, E_rev=0.0)
syn_IE_eval = build_synapse_block(pop_I_eval, pop_E_eval, W_IE; name=:syn_IE_eval, E_rev=-80.0)
syn_II_eval = build_synapse_block(pop_I_eval, pop_I_eval, W_II; name=:syn_II_eval, E_rev=-80.0)
synapse_specs_eval = [syn_EE_eval, syn_EI_eval, syn_IE_eval, syn_II_eval]

eval_net = build_acausal_network([pop_E_eval, pop_I_eval]; synapse_specs=synapse_specs_eval, drivers=drivers, name=:eval_net)
eval_sys = mtkcompile(eval_net.sys)
eval_prob = ODEProblem(eval_sys, [], (0.0, 50.0), jac=true, sparse=true)
eval_sol = solve(eval_prob, Rosenbrock23(); saveat=timesteps);

# --- Extract the free-running matrices for plotting ---
V_init_E_mat = reduce(hcat, init_sol[init_sys.pop_E_init.cap.v])
V_init_I_mat = reduce(hcat, init_sol[init_sys.pop_I_init.cap.v])
V_eval_E_mat = reduce(hcat, eval_sol[eval_sys.pop_E_eval.cap.v])
V_eval_I_mat = reduce(hcat, eval_sol[eval_sys.pop_I_eval.cap.v])

# --- Plotting the Free-Running Dynamics ---
subset_E = [1, 5, 10]
labels_E_true = ["True E" "" ""]
labels_E_init = ["Init E" "" ""]
labels_E_fit  = ["Fit E" "" ""]

p1 = plot(timesteps, V_data_E_mat[subset_E,:]', color=:black, lw=2, label=labels_E_true)
plot!(p1, timesteps, V_init_E_mat[subset_E,:]', color=:steelblue, lw=1.5, label=labels_E_init)
plot!(p1, timesteps, V_eval_E_mat[subset_E,:]', color=:crimson, lw=1.5, label=labels_E_fit)
title!("Excitatory Population (Arbitrary Subset)")

labels_I_true = ["True I" "" ""]
labels_I_init = ["Init I" "" ""]
labels_I_fit  = ["Fit I" "" ""]

p2 = plot(timesteps, V_data_I_mat', color=:black, lw=2, label=labels_I_true)
plot!(p2, timesteps, V_init_I_mat', color=:steelblue, lw=1.5, label=labels_I_init)
plot!(p2, timesteps, V_eval_I_mat', color=:crimson, lw=1.5, label=labels_I_fit)
title!("Inhibitory Population")

# Add initial guesses to these two plots
p3 = plot(1:N_E, gNa_E_true, label="True E gNa", lw=2, color=:black, shape=:circle)
plot!(p3, 1:N_E, guess_gNa_E, label="Init E gNa", lw=2, color=:steelblue, shape=:utriangle)
plot!(p3, 1:N_E, opt_gNa_E, label="Recovered E gNa", lw=2, color=:crimson, shape=:square)
title!("Heterogeneous gNa Recovery (E Pop)")

p4 = plot(1:N_I, gNa_I_true, label="True I gNa", lw=2, color=:black, shape=:circle)
plot!(p4, 1:N_I, guess_gNa_I, label="Init I gNa", lw=2, color=:steelblue, shape=:utriangle)
plot!(p4, 1:N_I, opt_gNa_I, label="Recovered I gNa", lw=2, color=:crimson, shape=:square)
title!("Heterogeneous gNa Recovery (I Pop)")

p = plot(p1, p2, p3, p4, layout=(4,1), size=(900, 1000), legend=:outertop)
xlabel!(p, "Time (ms) / Neuron Index")
ylabel!(p, "V (mV) / Conductance")

p

# ## 5. Comparison of scalar parameters

Markdown.parse("""
| Parameter | True Value | Initial Guess | Recovered Value |
|-----------|------------|---------------|-----------------|
| E: gK     | $true_gK   | $guess_gK     | $(round(opt_gK_E, digits=3)) |
| E: gleak  | $true_gleak| $guess_gleak  | $(round(opt_gleak_E, digits=3)) |
| I: gK     | $true_gK   | $guess_gK     | $(round(opt_gK_I, digits=3)) |
| I: gleak  | $true_gleak| $guess_gleak  | $(round(opt_gleak_I, digits=3)) |
""")

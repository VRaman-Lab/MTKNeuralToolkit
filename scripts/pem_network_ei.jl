using MTKNeuralToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, Pre
using SymbolicIndexingInterface: getu
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named, @component, @parameters, @unpack, Equation, System, extend
using Symbolics: SymbolicT
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using SciMLStructures: Tunable, canonicalize, replace
using SymbolicIndexingInterface: parameter_values, setp
using PreallocationTools
using DataInterpolations
using SciMLBase
using Plots

# ==========================================
# Custom Vectorized PEM Observer
# ==========================================
# To avoid MTK array-function broadcasting issues, we explicitly construct 
# the target vector from an array of scalar interpolations.
@component function VecPEMObservationChannel(; name, itps, K_init=1.0, topology=Vectorized(1))
    N = topology.N
    @named oneport = VectorizedOnePort(N=N)
    @unpack v, i = oneport
    
    @parameters K = K_init
    params = SymbolicT[K]
    vars = SymbolicT[]
    
    # Create a symbolic vector by evaluating each scalar interpolation
    target_vec = SymbolicT[itps[j](t) for j in 1:N]
    
    eqs = Equation[
        i ~ K .* (v .- target_vec)
    ]
    
    return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
end

# ==========================================
# 1. Build the True System & Generate Data
# ==========================================
N_E = 2
N_I = 1
top_E = Vectorized(N_E)
top_I = Vectorized(N_I)

function build_population(name::Symbol, top; gNa=120.0, gK=36.0, pem=false, itps=nothing, K=1.0)
    @named cap  = Capacitor(topology=top, C=1.0)
    @named na   = SodiumChannel(topology=top, g=gNa)
    @named k    = PotassiumChannel(topology=top, g=gK)
    @named leak = LeakChannel(topology=top)
    
    channels = [na, k, leak]
    if pem
        @named pem_ch = VecPEMObservationChannel(itps=itps, K_init=K, topology=top)
        push!(channels, pem_ch)
    end
    
    return build_compartment(cap, channels; name=name, V_init=-65.0, topology=top)
end

true_gNa = 120.0
true_gK  = 36.0

pop_E_true = build_population(:pop_E_true, top_E; gNa=true_gNa, gK=true_gK)
pop_I_true = build_population(:pop_I_true, top_I; gNa=true_gNa, gK=true_gK)

# Define simple dense weight matrices (N_post x N_pre)
W_EE = 0.5 .* ones(N_E, N_E)
W_EI = 1.0 .* ones(N_I, N_E) # E -> I
W_IE = 2.0 .* ones(N_E, N_I) # I -> E
W_II = 1.0 .* ones(N_I, N_I)

# Build synapse blocks
syn_EE = build_synapse_block(pop_E_true, pop_E_true, W_EE; name=:syn_EE, E_rev=0.0)
syn_EI = build_synapse_block(pop_E_true, pop_I_true, W_EI; name=:syn_EI, E_rev=0.0)
syn_IE = build_synapse_block(pop_I_true, pop_E_true, W_IE; name=:syn_IE, E_rev=-80.0)
syn_II = build_synapse_block(pop_I_true, pop_I_true, W_II; name=:syn_II, E_rev=-80.0)

synapse_specs = [syn_EE, syn_EI, syn_IE, syn_II]
drivers = [(1, 15.0)] # Drive the E population

true_net = build_acausal_network([pop_E_true, pop_I_true]; synapse_specs=synapse_specs, drivers=drivers, name=:true_net)

println("Compiling true vectorized network...")
true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, 50.0), jac=true, sparse=true)

timesteps = 0.0:0.1:50.0
println("Generating training data...")
true_sol = solve(true_prob, Tsit5(); saveat=timesteps)

# Extract true voltage traces as matrices (Neurons x Time)
V_data_E_mat = reduce(hcat, true_sol[true_sys.pop_E_true.cap.v]) 
V_data_I_mat = reduce(hcat, true_sol[true_sys.pop_I_true.cap.v]) 

# Create an array of scalar interpolations for each neuron in the populations
itps_E = [LinearInterpolation(V_data_E_mat[i, :], timesteps) for i in 1:N_E]
itps_I = [LinearInterpolation(V_data_I_mat[i, :], timesteps) for i in 1:N_I]

# ==========================================
# 2. Setup the PEM Optimization Problem
# ==========================================
guess_gNa = 20.0
guess_gK  = 10.0

# Build fit networks with PEM attached
pop_E_fit = build_population(:pop_E_fit, top_E; gNa=guess_gNa, gK=guess_gK, pem=true, itps=itps_E, K=2.0)
pop_I_fit = build_population(:pop_I_fit, top_I; gNa=guess_gNa, gK=guess_gK, pem=true, itps=itps_I, K=2.0)

# Recreate synapse blocks for the fit network
syn_EE_fit = build_synapse_block(pop_E_fit, pop_E_fit, W_EE; name=:syn_EE_fit, E_rev=0.0)
syn_EI_fit = build_synapse_block(pop_E_fit, pop_I_fit, W_EI; name=:syn_EI_fit, E_rev=0.0)
syn_IE_fit = build_synapse_block(pop_I_fit, pop_E_fit, W_IE; name=:syn_IE_fit, E_rev=-80.0)
syn_II_fit = build_synapse_block(pop_I_fit, pop_I_fit, W_II; name=:syn_II_fit, E_rev=-80.0)

synapse_specs_fit = [syn_EE_fit, syn_EI_fit, syn_IE_fit, syn_II_fit]

fit_net = build_acausal_network([pop_E_fit, pop_I_fit]; synapse_specs=synapse_specs_fit, drivers=drivers, name=:fit_net)
fit_sys = mtkcompile(fit_net.sys)
fit_prob = ODEProblem(fit_sys, [], (0.0, 50.0), jac=true, sparse=true)

# Extract parameters to fit (gNa for both populations)
gNa_E_sym = fit_sys.pop_E_fit.na.g
gNa_I_sym = fit_sys.pop_I_fit.na.g

setter = setp(fit_prob, [gNa_E_sym, gNa_I_sym])
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))

# Getters for full vectorized states
v_getter_E = getu(fit_prob, fit_sys.pop_E_fit.cap.v)
v_getter_I = getu(fit_prob, fit_sys.pop_I_fit.cap.v)

# ==========================================
# 3. Define Loss Function & Optimize
# ==========================================
function loss(x, p)
    prob, timesteps, V_data_E, V_data_I, setter, diffcache, v_getter_E, v_getter_I = p
    
    ps = parameter_values(prob)
    buffer = get_tmp(diffcache, x)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps = replace(Tunable(), ps, buffer)
    
    setter(ps, x)
    newprob = remake(prob; p=ps)
    
    sol = solve(newprob, Tsit5(); saveat=timesteps)
    
    if !SciMLBase.successful_retcode(sol.retcode)
        return Inf
    end
    
    # Convert array-of-arrays back to matrix for comparison
    V_fit_E = reduce(hcat, v_getter_E(sol))
    V_fit_I = reduce(hcat, v_getter_I(sol))
    
    return (sum(abs2, V_fit_E .- V_data_E) + sum(abs2, V_fit_I .- V_data_I)) / (size(V_data_E, 2) * (N_E + N_I))
end

opt_params = (fit_prob, timesteps, V_data_E_mat, V_data_I_mat, setter, diffcache, v_getter_E, v_getter_I)
adtype = AutoForwardDiff()
optfn = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optfn, [guess_gNa, guess_gNa], opt_params)

# ==========================================
# 4. Optimize and Plot
# ==========================================
println("Solving with initial guesses for visualization...")
init_sol = solve(fit_prob, Tsit5(); saveat=timesteps)

println("Starting optimization...")
res = solve(optprob, BFGS(); maxiters=500)

println("True conductances: gNa = $true_gNa")
println("Recovered E conductance: gNa = $(res.u[1])")
println("Recovered I conductance: gNa = $(res.u[2])")

# Solve one final time with the optimized parameters to plot the fit
opt_ps = parameter_values(fit_prob)
opt_buffer = copy(canonicalize(Tunable(), opt_ps)[1])
opt_ps = replace(Tunable(), opt_ps, opt_buffer)
setter(opt_ps, res.u)
opt_prob_final = remake(fit_prob; p=opt_ps)

opt_sol = solve(opt_prob_final, Tsit5(); saveat=timesteps)

# Plotting E population
p1 = plot(timesteps, V_data_E_mat', label=["True E1" "True E2"], lw=2, color=[:black :darkgray])
plot!(p1, timesteps, reduce(hcat, init_sol[fit_sys.pop_E_fit.cap.v])', label="Init E", ls=:dot, lw=2, color=:gray)
plot!(p1, timesteps, reduce(hcat, opt_sol[fit_sys.pop_E_fit.cap.v])', label=["Fit E1" "Fit E2"], ls=:dash, lw=2, color=[:red :blue])
title!(p1, "Excitatory Population")

# Plotting I population
p2 = plot(timesteps, V_data_I_mat', label="True I", lw=2, color=:black)
plot!(p2, timesteps, reduce(hcat, init_sol[fit_sys.pop_I_fit.cap.v])', label="Init I", ls=:dot, lw=2, color=:gray)
plot!(p2, timesteps, reduce(hcat, opt_sol[fit_sys.pop_I_fit.cap.v])', label="Fit I", ls=:dash, lw=2, color=:red)
title!(p2, "Inhibitory Population")

final_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
xlabel!("Time (ms)")
ylabel!("V (mV)")

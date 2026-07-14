using MTKNeuralToolkit
using SymbolicIndexingInterface: getu
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named
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
# 1. Build the True System & Generate Data
# ==========================================
top = Vectorized(1)

function build_hh_vec_neuron(name::Symbol; gNa=120.0, gK=36.0, pem=false, itp=nothing, K=1.0)
    @named cap  = Capacitor(topology=top, C=1.0)
    @named na   = SodiumChannel(topology=top, g=gNa)
    @named k    = PotassiumChannel(topology=top, g=gK)
    @named leak = LeakChannel(topology=top)
    
    channels = [na, k, leak]
    if pem
        @named pem_ch = PEMObservationChannel(itp=itp, K_init=K, topology=top)
        push!(channels, pem_ch)
    end
    
    return build_compartment(cap, channels; name=name, V_init=-65.0, topology=top)
end

true_gNa = 120.0
true_gK  = 36.0

# Build true network: A drives B via a synapse
A_true = build_hh_vec_neuron(:A_true; gNa=true_gNa, gK=true_gK)
B_true = build_hh_vec_neuron(:B_true; gNa=true_gNa, gK=true_gK)

@named syn_AB = ExpSynapse(g_max=3.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

synapse_specs = [
    SynapseSpec(A_true.interfaces.V[1], B_true.interfaces.V[1], B_true.interfaces.I_syn[1], syn_AB)
]

drivers = [(1, 10.0)] 

true_net = build_acausal_network([A_true, B_true]; synapse_specs=synapse_specs, drivers=drivers, name=:true_net)

println("Compiling true vectorized system...")
true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, 100.0))

timesteps = 0.0:0.1:100.0
println("Generating training data...")
true_sol = solve(true_prob, Tsit5(); saveat=timesteps)

# Extract both true voltage traces
V_data_A = true_sol[true_sys.A_true.cap.v[1]]
V_data_B = true_sol[true_sys.B_true.cap.v[1]]

# Create interpolation objects for PEM on BOTH neurons
itp_A = LinearInterpolation(V_data_A, timesteps)
itp_B = LinearInterpolation(V_data_B, timesteps)

# ==========================================
# 2. Setup the PEM Optimization Problem
# ==========================================
guess_gNa = 20.0
guess_gK  = 10.0

# Build the fit network with PEM observer attached to BOTH A_fit and B_fit
A_fit = build_hh_vec_neuron(:A_fit; gNa=guess_gNa, gK=guess_gK, pem=true, itp=itp_A, K=2.0)
B_fit = build_hh_vec_neuron(:B_fit; gNa=guess_gNa, gK=guess_gK, pem=true, itp=itp_B, K=2.0)

@named syn_AB_fit = ExpSynapse(g_max=3.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)

synapse_specs_fit = [
    SynapseSpec(A_fit.interfaces.V[1], B_fit.interfaces.V[1], B_fit.interfaces.I_syn[1], syn_AB_fit)
]

fit_net = build_acausal_network([A_fit, B_fit]; synapse_specs=synapse_specs_fit, drivers=drivers, name=:fit_net)
fit_sys = mtkcompile(fit_net.sys)
fit_prob = ODEProblem(fit_sys, [], (0.0, 100.0))

# Extract all 4 symbolic parameters we want to fit
gNa_A_sym = fit_sys.A_fit.na.g
gK_A_sym  = fit_sys.A_fit.k.g
gNa_B_sym = fit_sys.B_fit.na.g
gK_B_sym  = fit_sys.B_fit.k.g

# High-performance parameter setter
setter = setp(fit_prob, [gNa_A_sym, gK_A_sym, gNa_B_sym, gK_B_sym])
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))

# Get the scalar voltages of both vectorized compartments
v_getter_A = getu(fit_prob, fit_sys.A_fit.cap.v[1])
v_getter_B = getu(fit_prob, fit_sys.B_fit.cap.v[1])

# ==========================================
# 3. Define Loss Function & Optimize
# ==========================================
function loss(x, p)
    prob, timesteps, V_data_A, V_data_B, setter, diffcache, v_getter_A, v_getter_B = p
    
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
    
    V_fit_A = v_getter_A(sol)
    V_fit_B = v_getter_B(sol)
    
    # Mean squared error across BOTH traces
    return (sum(abs2, V_fit_A .- V_data_A) + sum(abs2, V_fit_B .- V_data_B)) / (length(V_data_A) + length(V_data_B))
end

opt_params = (fit_prob, timesteps, V_data_A, V_data_B, setter, diffcache, v_getter_A, v_getter_B)
adtype = AutoForwardDiff()
optfn = OptimizationFunction(loss, adtype)
# We are now optimizing 4 parameters: [gNa_A, gK_A, gNa_B, gK_B]
optprob = OptimizationProblem(optfn, [guess_gNa, guess_gK, guess_gNa, guess_gK], opt_params)

# ==========================================
# 4. Optimize and Plot
# ==========================================
println("Solving with initial guesses for visualization...")
init_sol = solve(fit_prob, Tsit5(); saveat=timesteps)

println("Starting optimization...")
res = solve(optprob, BFGS(); maxiters=500)

println("True conductances: gNa = $true_gNa, gK = $true_gK")
println("Recovered A conductances: gNa = $(res.u[1]), gK = $(res.u[2])")
println("Recovered B conductances: gNa = $(res.u[3]), gK = $(res.u[4])")

# Solve one final time with the optimized parameters to plot the fit
opt_ps = parameter_values(fit_prob)
opt_buffer = copy(canonicalize(Tunable(), opt_ps)[1])
opt_ps = replace(Tunable(), opt_ps, opt_buffer)
setter(opt_ps, res.u)
opt_prob_final = remake(fit_prob; p=opt_ps)

opt_sol = solve(opt_prob_final, Tsit5(); saveat=timesteps)

p1 = plot(timesteps, V_data_A, label="True A", lw=2, color=:black)
plot!(p1, timesteps, init_sol[fit_sys.A_fit.cap.v[1]], label="Init A", ls=:dot, lw=2, color=:gray)
plot!(p1, timesteps, opt_sol[fit_sys.A_fit.cap.v[1]], label="Fit A", ls=:dash, lw=2, color=:red)
title!(p1, "Presynaptic (Driven) Neuron")

p2 = plot(timesteps, V_data_B, label="True B", lw=2, color=:black)
plot!(p2, timesteps, init_sol[fit_sys.B_fit.cap.v[1]], label="Init B", ls=:dot, lw=2, color=:gray)
plot!(p2, timesteps, opt_sol[fit_sys.B_fit.cap.v[1]], label="Fit B", ls=:dash, lw=2, color=:red)
title!(p2, "Postsynaptic Neuron")

final_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
xlabel!("Time (ms)")
ylabel!("V (mV)")

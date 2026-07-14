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
top = Scalar()

function build_hh_neuron(name::Symbol; gNa=120.0, gK=36.0, pem=false, itp=nothing, K=1.0)
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

true_neuron = build_hh_neuron(:true_neuron; gNa=true_gNa, gK=true_gK)
drivers = [(1, 10.0)] 
true_net = build_acausal_network([true_neuron]; drivers=drivers, name=:true_net)

println("Compiling true system...")
true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, 100.0))

timesteps = 0.0:0.1:100.0
println("Generating training data...")
true_sol = solve(true_prob, Tsit5(); saveat=timesteps)
V_data = true_sol[true_sys.true_neuron.cap.v]

# Create the interpolation object for PEM
itp = LinearInterpolation(V_data, timesteps)

# ==========================================
# 2. Setup the PEM Optimization Problem
# ==========================================
guess_gNa = 20.0
guess_gK  = 10.0

# Build the fit network with PEM observer attached (K fixed at 2.0)
fit_neuron = build_hh_neuron(:fit_neuron; gNa=guess_gNa, gK=guess_gK, pem=true, itp=itp, K=2.0)
fit_net = build_acausal_network([fit_neuron]; drivers=drivers, name=:fit_net)
fit_sys = mtkcompile(fit_net.sys)
fit_prob = ODEProblem(fit_sys, [], (0.0, 100.0))

# Extract symbolic parameters we want to fit
gNa_sym = fit_sys.fit_neuron.na.g
gK_sym  = fit_sys.fit_neuron.k.g

# High-performance parameter setter and state getter
setter = setp(fit_prob, [gNa_sym, gK_sym])
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))

# Create a fast getter for the voltage state we want to compare
v_sym = fit_sys.fit_neuron.cap.v
v_getter = getu(fit_prob, v_sym)

# ==========================================
# 3. Define Loss Function & Optimize
# ==========================================
function loss(x, p)
    prob, timesteps, V_data, setter, diffcache, v_getter = p
    
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
    
    # Use our fast getter to extract the voltage
    V_fit = v_getter(sol)
    return sum(abs2, V_fit .- V_data) / length(V_data)
end

opt_params = (fit_prob, timesteps, V_data, setter, diffcache, v_getter)
adtype = AutoForwardDiff()
optfn = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optfn, [guess_gNa, guess_gK], opt_params)

# ==========================================
# 4. Optimize and Plot
# ==========================================
println("Solving with initial guesses for visualization...")
init_sol = solve(fit_prob, Tsit5(); saveat=timesteps)

println("Starting optimization...")
res = solve(optprob, BFGS(); maxiters=300)

println("True conductances: gNa = $true_gNa, gK = $true_gK")
println("Recovered conductances: gNa = $(res.u[1]), gK = $(res.u[2])")

# Solve one final time with the optimized parameters to plot the fit
opt_ps = parameter_values(fit_prob)
opt_buffer = copy(canonicalize(Tunable(), opt_ps)[1])
opt_ps = replace(Tunable(), opt_ps, opt_buffer)
setter(opt_ps, res.u)
opt_prob_final = remake(fit_prob; p=opt_ps)

opt_sol = solve(opt_prob_final, Tsit5(); saveat=timesteps)

p1 = plot(timesteps, V_data, label="True Voltage", lw=2, color=:black)
plot!(p1, timesteps, init_sol[fit_sys.fit_neuron.cap.v], label="Initial Guess", ls=:dot, lw=2, color=:gray)
plot!(p1, timesteps, opt_sol[fit_sys.fit_neuron.cap.v], label="Fitted Voltage", ls=:dash, lw=2, color=:red)
title!(p1, "PEM Parameter Estimation of HH Conductances")
xlabel!("Time (ms)")
ylabel!("V (mV)")
p1

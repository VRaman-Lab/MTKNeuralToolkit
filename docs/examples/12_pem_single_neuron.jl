# # **12.** Parameter Estimation: Single Neuron
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/15_pem_noisy_data.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/15_pem_noisy_data.ipynb)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Introduction
# In this example, we demonstrate the filtering power of the Prediction Error Method (PEM). 
# We generate data from a true Hodgkin-Huxley neuron, add Gaussian noise to the voltage trace, 
# and then attempt to recover the conductances. 
# The PEM observer channel acts like a fixed gain Kalman Filter, absorbing the noise while allowing the optimiser 
# to find the true underlying parameters without overfitting the noise.

#nb # %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
using MTKNeuralToolkit
using SymbolicIndexingInterface: getu
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
using OrdinaryDiffEqRosenbrock
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

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 1. Build the True System & Generate Noisy Data

#nb # %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
top = Scalar()

function build_hh_neuron(name::Symbol; gNa=120.0, gK=36.0, gleak=0.3, pem=false, itps=nothing, K=1.0)
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

true_gNa = 120.0
true_gK  = 36.0
true_gleak = 0.3

true_neuron = build_hh_neuron(:true_neuron; gNa=true_gNa, gK=true_gK, gleak=true_gleak)

drivers = [(1, 8.0)]
true_net = build_acausal_network([true_neuron]; drivers=drivers, name=:true_net)
END_TIME= 100.0


true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, END_TIME))
timesteps = 0.0:0.1:END_TIME

true_sol = solve(true_prob, Rodas5(); saveat=timesteps)
V_data_clean = true_sol[true_sys.true_neuron.cap.v]

# Add 3 mV Gaussian noise to the data
Random.seed!(42)
noise_level = 3.0 
V_data_noisy = V_data_clean .+ noise_level .* randn(length(V_data_clean))
itp_V = LinearInterpolation(V_data_noisy, timesteps)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 2. Setup the PEM Optimization Problem
# We create a model neuron with terrible initial guesses and attach a PEM observer to the noisy data.

#nb # %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
guess_gNa = 10.0
guess_gK  = 100.0
guess_gleak = 10.3

fit_neuron = build_hh_neuron(:fit_neuron; gNa=guess_gNa, gK=guess_gK, gleak=guess_gleak, pem=true, itps=[itp_V], K=2.0)
fit_net = build_acausal_network([fit_neuron]; drivers=drivers, name=:fit_net)
fit_sys = mtkcompile(fit_net.sys)
fit_prob = ODEProblem(fit_sys, [], (0.0, END_TIME))

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
gNa_sym = fit_sys.fit_neuron.na.g
gK_sym  = fit_sys.fit_neuron.k.g
gleak_sym = fit_sys.fit_neuron.leak.g

setter = setp(fit_prob, [gNa_sym, gK_sym, gleak_sym])
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))

v_getter = getu(fit_prob, fit_sys.fit_neuron.cap.v)
i_pem_getter = getu(fit_prob, fit_sys.fit_neuron.pem_ch.i)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 3. Define Loss Function & Optimize
# We calculate the loss based on the *PEM observer current*. 
# Instead of minimizing the tracking error (which the controller achieves regardless of parameters), 
# we minimize the effort of the controller. This forces the optimizer to find the underlying parameters 
# that naturally generate the data.

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 3. Define Loss Function & Optimize
# We calculate a multi-objective loss based on both the tracking error (voltage) 
# and the observer effort (current). This forces the system to track the data 
# while penalizing the controller from "cheating" to force bad parameters to fit.

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
function loss(x, p)
    # Unpack 7 items:
    prob, timesteps, V_data_noisy, setter, diffcache, v_getter, i_pem_getter = p
    
    ps = parameter_values(prob)
    buffer = get_tmp(diffcache, x)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps = replace(Tunable(), ps, buffer)
    setter(ps, x)
    newprob = remake(prob; p=ps)
    sol = solve(newprob, Rodas5(); saveat=timesteps, reltol=1e-8, abstol=1e-8)
    
    if !SciMLBase.successful_retcode(sol.retcode)
        return Inf
    end
    
    V_fit = v_getter(sol)
    I_pem = i_pem_getter(sol)
    
    # Multi-objective cost
    tracking_error = sum(abs2, V_fit .- V_data_noisy) / length(V_data_noisy)
    observer_effort = sum(abs2, I_pem) / length(I_pem)
    
    # Weights (tune these in general, although this simulation doesn't care about them)
    alpha = 1.0  
    beta = 1.0   
    
    return alpha * tracking_error + beta * observer_effort
end

# Tuple must also have exactly 7 items:
opt_params = (fit_prob, timesteps, V_data_noisy, setter, diffcache, v_getter, i_pem_getter)
adtype = AutoForwardDiff()
optfn = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optfn, [guess_gNa, guess_gK, guess_gleak], opt_params)



#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 4. Optimize and Plot
# To avoid recompiling new systems for the free-running simulations, we simply reuse 
# the compiled `fit_sys` and set the PEM controller gain (K) to 0.0 to disable the observer.

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
println("Starting optimization...")
res = solve(optprob, BFGS(); maxiters=1000)

K_sym = fit_sys.fit_neuron.pem_ch.K
K_setter = setp(fit_prob, [K_sym])

# --- 1. Simulate with recovered parameters (WITH PEM) to get observer current ---
opt_ps = parameter_values(fit_prob)
opt_buffer = copy(canonicalize(Tunable(), opt_ps)[1])
opt_ps = replace(Tunable(), opt_ps, opt_buffer)
setter(opt_ps, res.u) # set recovered parameters
opt_prob_pem = remake(fit_prob; p=opt_ps)
opt_sol_pem = solve(opt_prob_pem, Rodas5(); saveat=timesteps, reltol=1e-6, abstol=1e-6)

# --- 2. Pure simulation for INITIAL GUESS (No PEM) ---
init_ps = parameter_values(fit_prob)
init_buffer = copy(canonicalize(Tunable(), init_ps)[1])
init_ps = replace(Tunable(), init_ps, init_buffer)
setter(init_ps, [guess_gNa, guess_gK, guess_gleak])
K_setter(init_ps, [0.0]) # disable PEM controller
init_prob_free = remake(fit_prob; p=init_ps)
init_eval_sol = solve(init_prob_free, Rodas5(); saveat=timesteps)

# --- 3. Pure simulation for RECOVERED PARAMETERS (No PEM) ---
opt_gNa, opt_gK, opt_gleak = res.u
fit_ps = parameter_values(fit_prob)
fit_buffer = copy(canonicalize(Tunable(), fit_ps)[1])
fit_ps = replace(Tunable(), fit_ps, fit_buffer)
setter(fit_ps, [opt_gNa, opt_gK, opt_gleak])
# disable PEM controller
K_setter(fit_ps, [0.0]) 
fit_prob_free = remake(fit_prob; p=fit_ps)
fit_eval_sol = solve(fit_prob_free, Rodas5(); saveat=timesteps)

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
# (Plotting code remains exactly the same)
p1 = plot(timesteps, V_data_noisy, label="Noisy Target", color=:gray, lw=1, alpha=0.8)
plot!(p1, timesteps, V_data_clean, label="True Clean", color=:black, lw=2)
plot!(p1, timesteps, init_eval_sol[fit_sys.fit_neuron.cap.v], label="Initial Guess (Free-Running)", ls=:dot, lw=2, color=:blue)
plot!(p1, timesteps, fit_eval_sol[fit_sys.fit_neuron.cap.v], label="Fit (Free-Running)", ls=:dash, lw=2, color=:red)
title!("Voltage Trace Recovery (Noisy Data)")

# Plot the observer current. 
I_obs = i_pem_getter(opt_sol_pem)
p2 = plot(timesteps, I_obs, label="Observer Current", color=:red, lw=1.5)
title!("PEM Observer Current (Absorbing Noise)")
hline!([0.0], color=:gray, ls=:dot, label="0")

p = plot(p1, p2, layout=(2,1), size=(800, 700), legend=:outertop)
xlabel!(p, "Time (ms)")
ylabel!(p, "V (mV) / I (nA)")
p
# Note the slight phase lag at the end of the sim.

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 5. Parameter Comparison

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
Markdown.parse("""
| Parameter | True Value | Initial Guess | Recovered Value |
|-----------|------------|---------------|-----------------|
| gNa       | $true_gNa  | $guess_gNa    | $(round(opt_gNa, digits=3)) |
| gK        | $true_gK   | $guess_gK     | $(round(opt_gK, digits=3)) |
| gleak     | $true_gleak| $guess_gleak  | $(round(opt_gleak, digits=3)) |
""")



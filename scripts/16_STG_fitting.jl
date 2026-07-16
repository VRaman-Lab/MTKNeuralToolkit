# # **16.** Parameter Estimation: STG Synaptic Weights
#
#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/generated/16_pem_stg_synapses.ipynb)
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/16_pem_stg_synapses.ipynb)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Introduction
# In this example, we attempt parameter estimation on the canonical 3-neuron Stomatogastric Ganglion (STG) circuit.
# We keep the intrinsic conductances fixed at their true values and attempt to recover the weights of the 7 chemical synapses.
# We use a 5-second simulation window to capture multiple bursts of the slow pyloric rhythm.

#nb # %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron: PrinzGeometry, na_gates, cas_gates, cat_gates, ka_gates, kca_gates, kdr_gates, h_gates
using MTKNeuralToolkit.HodgkinHuxley: SodiumChannel, PotassiumChannel, LeakChannel
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq
import OrdinaryDiffEqSDIRK: TRBDF2
using Optimization
using OptimizationOptimJL
using SciMLStructures: Tunable, canonicalize, replace
using SymbolicIndexingInterface: parameter_values, setp, getu
using PreallocationTools
using DataInterpolations
using SciMLBase
using Plots
using Markdown

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 1. Build the True STG System
# We replicate the exact build process from the PrinzNeuron library to generate our target data.

#nb # %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
geom = PrinzGeometry(area=0.0628, C_m=10.0)
tauCa = 200.0
Ca_inf = 0.05
nernst_factor = 500.0 * 8.6174e-5 * 283.15
prinz_ion_config = CalciumTracker(decay=ca -> (Ca_inf .- ca) ./ tauCa, Ca_init=Ca_inf)

# Local Channel Builders
NaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=50.0, gates=na_gates, geometry=geom)
CaSCh(g; name)  = CaVChannel(name=name, g=g, gates=cas_gates, Ca_out=3000.0, nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
CaTCh(g; name)  = CaVChannel(name=name, g=g, gates=cat_gates, Ca_out=3000.0, nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
HCh(g; name)    = GenericChannel(name=name, g=g, E_rev=-20.0, gates=h_gates, geometry=geom)
KaCh(g; name)   = GenericChannel(name=name, g=g, E_rev=-80.0, gates=ka_gates, geometry=geom)
KCaCh(g; name)  = KCaChannel(name=name, g=g, E_rev=-80.0, gates=kca_gates, geometry=geom)
KdrCh(g; name)  = GenericChannel(name=name, g=g, E_rev=-80.0, gates=kdr_gates, geometry=geom)
LeakCh(g; name) = GenericChannel(name=name, g=g, E_rev=-50.0, gates=GateSpec[], geometry=geom)

function build_stg_true(; name=:stg_true)
    function build_AB()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(6.0);  @named cat = CaTCh(2.5)
        @named h    = HCh(0.01);   @named ka   = KaCh(50.0);  @named kca = KCaCh(5.0)
        @named kdr  = KdrCh(100.0)
        return build_compartment(cap, [na, cas, cat, h, ka, kca, kdr]; name=:AB, V_init=-60.0, ion_config=prinz_ion_config)
    end

    function build_PY()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(2.0);  @named cat = CaTCh(2.4)
        @named h    = HCh(0.05);   @named ka   = KaCh(50.0);  @named kdr = KdrCh(125.0)
        @named leak = LeakCh(0.01)
        return build_compartment(cap, [na, cas, cat, h, ka, kdr, leak]; name=:PY, V_init=-55.0, ion_config=prinz_ion_config)
    end

    function build_LP()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(4.0)
        @named h    = HCh(0.05);   @named ka   = KaCh(20.0);  @named kdr = KdrCh(25.0)
        @named leak = LeakCh(0.03)
        return build_compartment(cap, [na, cas, h, ka, kdr, leak]; name=:LP, V_init=-65.0, ion_config=prinz_ion_config)
    end

    AB = build_AB(); PY = build_PY(); LP = build_LP()
    neurons = [AB, PY, LP]

    @named ABLP_chol = CholSynapse(g_max=30.0, geometry=geom)
    @named ABPY_chol = CholSynapse(g_max=3.0 , geometry=geom)
    @named ABLP_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named ABPY_glut = GlutSynapse(g_max=10.0, geometry=geom)
    @named LPAB_glut = GlutSynapse(g_max=30.0, geometry=geom)
    @named LPPY_glut = GlutSynapse(g_max=1.0 , geometry=geom)
    @named PYLP_glut = GlutSynapse(g_max=30.0, geometry=geom)

    synapse_specs = [
        SynapseSpec(LP.interfaces.V, AB.interfaces.V, AB.interfaces.I_syn, LPAB_glut),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_chol),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_glut),
        SynapseSpec(LP.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, LPPY_glut),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_chol),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_glut),
        SynapseSpec(PY.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, PYLP_glut)
    ]

    return build_acausal_network(neurons; synapse_specs=synapse_specs, name=name)
end

true_net = build_stg_true()
true_sys = mtkcompile(true_net.sys)
true_prob = ODEProblem(true_sys, [], (0.0, 500.0), jac=true, sparse=true) # 5 seconds
timesteps = 0.0:1.0:500.0 # 10ms steps

println("Generating 5s of STG training data...")
true_sol = solve(true_prob, TRBDF2(); saveat=timesteps) 

V_data_AB = true_sol[true_sys.AB.cap.v]
V_data_LP = true_sol[true_sys.LP.cap.v]
V_data_PY = true_sol[true_sys.PY.cap.v]

itp_AB = LinearInterpolation(V_data_AB, timesteps)
itp_LP = LinearInterpolation(V_data_LP, timesteps)
itp_PY = LinearInterpolation(V_data_PY, timesteps)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 2. Setup the PEM Optimization Problem
# We rebuild the STG circuit, but this time we attach PEM observers to all 3 neurons.
# We intentionally guess terrible synapse weights (`g_max=1.0`) so the initial model's rhythm falls apart.

#nb # %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
function build_stg_fit(; name=:stg_fit)
    function build_AB()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(6.0);  @named cat = CaTCh(2.5)
        @named h    = HCh(0.01);   @named ka   = KaCh(50.0);  @named kca = KCaCh(5.0)
        @named kdr  = KdrCh(100.0)
        @named pem  = PEMObservationChannel(itps=[itp_AB], K_init=0.5, topology=Scalar())
        return build_compartment(cap, [na, cas, cat, h, ka, kca, kdr, pem]; name=:AB, V_init=-60.0, ion_config=prinz_ion_config)
    end

    function build_PY()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(2.0);  @named cat = CaTCh(2.4)
        @named h    = HCh(0.05);   @named ka   = KaCh(50.0);  @named kdr = KdrCh(125.0)
        @named leak = LeakCh(0.01)
        @named pem  = PEMObservationChannel(itps=[itp_PY], K_init=0.5, topology=Scalar())
        return build_compartment(cap, [na, cas, cat, h, ka, kdr, leak, pem]; name=:PY, V_init=-55.0, ion_config=prinz_ion_config)
    end

    function build_LP()
        @named cap  = Capacitor(geometry=geom)
        @named na   = NaCh(100.0); @named cas  = CaSCh(4.0)
        @named h    = HCh(0.05);   @named ka   = KaCh(20.0);  @named kdr = KdrCh(25.0)
        @named leak = LeakCh(0.03)
        @named pem  = PEMObservationChannel(itps=[itp_LP], K_init=0.5, topology=Scalar())
        return build_compartment(cap, [na, cas, h, ka, kdr, leak, pem]; name=:LP, V_init=-65.0, ion_config=prinz_ion_config)
    end

    AB = build_AB(); PY = build_PY(); LP = build_LP()
    neurons = [AB, PY, LP]

    # Terrible initial guesses (1.0 instead of 3.0-30.0)
    @named ABLP_chol = CholSynapse(g_max=1.0, geometry=geom)
    @named ABPY_chol = CholSynapse(g_max=1.0, geometry=geom)
    @named ABLP_glut = GlutSynapse(g_max=1.0, geometry=geom)
    @named ABPY_glut = GlutSynapse(g_max=1.0, geometry=geom)
    @named LPAB_glut = GlutSynapse(g_max=1.0, geometry=geom)
    @named LPPY_glut = GlutSynapse(g_max=1.0, geometry=geom)
    @named PYLP_glut = GlutSynapse(g_max=1.0, geometry=geom)

    synapse_specs = [
        SynapseSpec(LP.interfaces.V, AB.interfaces.V, AB.interfaces.I_syn, LPAB_glut),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_chol),
        SynapseSpec(AB.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, ABPY_glut),
        SynapseSpec(LP.interfaces.V, PY.interfaces.V, PY.interfaces.I_syn, LPPY_glut),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_chol),
        SynapseSpec(AB.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, ABLP_glut),
        SynapseSpec(PY.interfaces.V, LP.interfaces.V, LP.interfaces.I_syn, PYLP_glut)
    ]

    return build_acausal_network(neurons; synapse_specs=synapse_specs, name=name)
end

fit_net = build_stg_fit()
fit_sys = mtkcompile(fit_net.sys)
fit_prob = ODEProblem(fit_sys, [], (0.0, 500.0), jac=true, sparse=true)

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
# Extract the 7 synapse g_max symbols
syn_syms = [
    fit_sys.LPAB_glut.g_max, fit_sys.ABPY_chol.g_max, fit_sys.ABPY_glut.g_max,
    fit_sys.LPPY_glut.g_max, fit_sys.ABLP_chol.g_max, fit_sys.ABLP_glut.g_max,
    fit_sys.PYLP_glut.g_max
]

setter = setp(fit_prob, syn_syms)
diffcache = DiffCache(copy(canonicalize(Tunable(), parameter_values(fit_prob))[1]))

v_getter_AB = getu(fit_prob, fit_sys.AB.cap.v)
v_getter_LP = getu(fit_prob, fit_sys.LP.cap.v)
v_getter_PY = getu(fit_prob, fit_sys.PY.cap.v)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 3. Define Loss Function & Optimize
# We calculate the combined MSE across all 3 neurons against the true STG rhythm.

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
function loss(x, p)
    prob, timesteps, V_AB, V_LP, V_PY, setter, diffcache, g_AB, g_LP, g_PY = p
    ps = parameter_values(prob)
    buffer = get_tmp(diffcache, x)
    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps = replace(Tunable(), ps, buffer)
    setter(ps, x)
    newprob = remake(prob; p=ps)
    sol = solve(newprob, TRBDF2(); saveat=timesteps)
    if !SciMLBase.successful_retcode(sol.retcode)
        return Inf
    end
    err = sum(abs2, g_AB(sol) .- V_AB) + sum(abs2, g_LP(sol) .- V_LP) + sum(abs2, g_PY(sol) .- V_PY)
    return err / (3 * length(V_AB))
end

x0 = fill(10.0, 7) # 7 bad initial guesses
opt_params = (fit_prob, timesteps, V_data_AB, V_data_LP, V_data_PY, setter, diffcache, v_getter_AB, v_getter_LP, v_getter_PY)
adtype = AutoForwardDiff()
optfn = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optfn, x0, opt_params)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 4. Optimize and Plot

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
println("Solving with initial guesses for visualization...")
init_sol = solve(fit_prob, TRBDF2(); saveat=timesteps)

println("Starting optimization (7 synapses)...")
res = solve(optprob, BFGS(); maxiters=500)

# Apply optimized parameters
opt_ps = parameter_values(fit_prob)
opt_buffer = copy(canonicalize(Tunable(), opt_ps)[1])
opt_ps = replace(Tunable(), opt_ps, opt_buffer)
setter(opt_ps, res.u)
opt_prob_final = remake(fit_prob; p=opt_ps)
opt_sol = solve(opt_prob_final, TRBDF2(); saveat=timesteps)

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
p1 = plot(timesteps, V_data_AB, label="True AB", color=:black, lw=2)
plot!(p1, timesteps, init_sol[fit_sys.AB.cap.v], label="Init AB", color=:gray, ls=:dot)
plot!(p1, timesteps, opt_sol[fit_sys.AB.cap.v], label="Fit AB", color=:red, ls=:dash, lw=2)
title!("AB Neuron")

p2 = plot(timesteps, V_data_LP, label="True LP", color=:black, lw=2)
plot!(p2, timesteps, init_sol[fit_sys.LP.cap.v], label="Init LP", color=:gray, ls=:dot)
plot!(p2, timesteps, opt_sol[fit_sys.LP.cap.v], label="Fit LP", color=:blue, ls=:dash, lw=2)
title!("LP Neuron")

p3 = plot(timesteps, V_data_PY, label="True PY", color=:black, lw=2)
plot!(p3, timesteps, init_sol[fit_sys.PY.cap.v], label="Init PY", color=:gray, ls=:dot)
plot!(p3, timesteps, opt_sol[fit_sys.PY.cap.v], label="Fit PY", color=:green, ls=:dash, lw=2)
title!("PY Neuron")

plot(p1, p2, p3, layout=(3,1), size=(900, 800), legend=:outertop)
xlabel!("Time (ms)")
ylabel!("V (mV)")

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 5. Synapse Weight Comparison

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
true_syn = [30.0, 3.0, 10.0, 1.0, 30.0, 30.0, 30.0]
syn_names = ["LPAB", "ABPY_chol", "ABPY_glut", "LPPY", "ABLP_chol", "ABLP_glut", "PYLP"]

println("\n--- Synapse Weight Recovery ---")
for i in 1:7
    println("$(syn_names[i]): True = $(true_syn[i]), Recovered = $(round(res.u[i], digits=2))")
end

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# ## 6. Validation: Remove the Observer
# To prove the recovered synapses are correct (and the observer isn't just faking the fit), 
# we build a 3rd STG network with NO PEM observers, inject the optimized synapse weights, 
# and simulate. If the parameters are correct, the STG rhythm should persist naturally!

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
# We can just reuse the true network structure, but we need to update its synapse parameters.
# Since we already have true_net compiled, we can just update its synapses to the recovered values.
val_sys = true_sys
val_prob = ODEProblem(val_sys, [], (0.0, 500.0))

# Extract the true network's synapse symbols
val_syn_syms = [
    val_sys.LPAB_glut.g_max, val_sys.ABPY_chol.g_max, val_sys.ABPY_glut.g_max,
    val_sys.LPPY_glut.g_max, val_sys.ABLP_chol.g_max, val_sys.ABLP_glut.g_max,
    val_sys.PYLP_glut.g_max
]
val_setter = setp(val_prob, val_syn_syms)

# Apply the recovered parameters to the pure (observer-free) network
val_ps = parameter_values(val_prob)
val_buffer = copy(canonicalize(Tunable(), val_ps)[1])
val_ps = replace(Tunable(), val_ps, val_buffer)
val_setter(val_ps, res.u)

val_prob_final = remake(val_prob; p=val_ps)
val_sol = solve(val_prob_final, TRBDF2(); saveat=timesteps)

#nb %% A slide [code] {"slideshow": {"slide_type": "fragment"}}
p4 = plot(timesteps, V_data_AB, label="True AB", color=:black, lw=2)
plot!(p4, timesteps, val_sol[val_sys.AB.cap.v], label="Validated AB", color=:red, ls=:dash, lw=2)
title!("AB Neuron (No Observer)")

p5 = plot(timesteps, V_data_LP, label="True LP", color=:black, lw=2)
plot!(p5, timesteps, val_sol[val_sys.LP.cap.v], label="Validated LP", color=:blue, ls=:dash, lw=2)
title!("LP Neuron (No Observer)")

p6 = plot(timesteps, V_data_PY, label="True PY", color=:black, lw=2)
plot!(p6, timesteps, val_sol[val_sys.PY.cap.v], label="Validated PY", color=:green, ls=:dash, lw=2)
title!("PY Neuron (No Observer)")

plot(p4, p5, p6, layout=(3,1), size=(900, 800), legend=:outertop)
xlabel!("Time (ms)")
ylabel!("V (mV)")

using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron
using ModelingToolkit: mtkcompile, parameters, tunable_parameters
using OrdinaryDiffEq, Plots
using ForwardDiff
using PreallocationTools
using SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_values
using SciMLStructures: Tunable, canonicalize, replace
using SciMLBase, LinearAlgebra
using MinimallyDisruptiveCurves

# ==========================================
# 1. Setup MTK Network & Base ODE Problem
# ==========================================
comp = PrinzNeuron.build_prinz_neuron()
drivers = [(1, 10.0)] # Kick it off
net = build_acausal_network([comp]; drivers=drivers)
net_compiled = mtkcompile(net.sys)

tspan = (0.0, 5000.0) # Adjusted to 500s to match truth generation
prob = ODEProblem(net_compiled, [], tspan, jac=true, sparse=true)

# Overwrite with manually specified initial conditions to skip transients
u0_clean = [
    0.009107988226400918, 0.5790449103338455, 0.034749193876961876, 0.17576649080963375,
    0.05292650006369318, 0.46813340850278, 0.020714279904595786, 0.031565920507008835,
    0.8355516740569822, 0.0647638489154021, 0.32627808498550076, -50.27579272693781, 107.36460442520591
]
prob = remake(prob; u0 = u0_clean, tspan = tspan)

# ==========================================
# 2. Define the Specific Tracking States (Observables)
# ==========================================
target_observables = [
    net_compiled.Prinz_Neuron.Prinz_Neuron_ca_pool.Ca # Calcium
]

# ==========================================
# 3. Dynamic Parameter Identification (Maximal Conductances)
# ==========================================
# Explicitly select the conductance parameters based on the list provided
all_params = parameters(net_compiled)
params_to_optimize = filter(p -> occursin("₊g", string(p)), all_params)

println("Natively optimizing $(length(params_to_optimize)) maximal conductances:")
println.(string.(params_to_optimize))

# ==========================================
# 4. Generate Baseline Experimental Data ("Truth")
# ==========================================
timesteps = 0.0:2.0:5000.0
sol_nominal = solve(prob, Tsit5(); saveat = timesteps)
truth_data = Array(sol_nominal(timesteps, idxs = target_observables))

# Plot to verify clean start
p1 = plot(sol_nominal, idxs=[net_compiled.Prinz_Neuron.cap.v], title="Membrane Potential")
p2 = plot(sol_nominal, idxs=[net_compiled.Prinz_Neuron.Prinz_Neuron_ca_pool.Ca], title="Calcium")
display(plot(p1, p2, layout=(2,1)))


# ==========================================
# 5. High-Performance Loss Function
# ==========================================
function loss_function(x, p_tuple)
    odeprob, ts, truth, setter, diffcache, obs_symbols = p_tuple

    ps = parameter_values(odeprob)
    buffer = get_tmp(diffcache, x)

    copyto!(buffer, canonicalize(Tunable(), ps)[1])
    ps_updated = replace(Tunable(), ps, buffer)

    setter(ps_updated, x)

    newprob = remake(odeprob; p = ps_updated)
    # Using Tsit5 as specified for the neuron model
    sol = solve(newprob, Tsit5(); saveat = ts)

    if sol.retcode != SciMLBase.ReturnCode.Success
        return eltype(x)(Inf)
    end

    current_data = sol(ts, idxs = obs_symbols)
    return sum(abs2, truth .- current_data) / length(truth)
end

# ==========================================
# 6. Build the Optimization Context
# ==========================================
setter = setp(prob, params_to_optimize)
getter = getp(prob, params_to_optimize)

raw_ps = parameter_values(prob)
tunable_vector_prototype = copy(canonicalize(Tunable(), raw_ps)[1])
# diffcache = DiffCache(tunable_vector_prototype)
diffcache = DiffCache(tunable_vector_prototype, 80)


p_tuple = (prob, timesteps, truth_data, setter, diffcache, target_observables)

# ==========================================
# 7. Evaluation and Gradient Verification
# ==========================================
println("\n--- Running Scaled Loss Function Evaluation ---")

x_nominal = getter(prob)
loss_at_nominal = loss_function(x_nominal, p_tuple)
println("Loss at nominal: $loss_at_nominal")

f_wrapped = θ -> loss_function(θ, p_tuple)

# Pre-allocate the ForwardDiff configuration
cfg = ForwardDiff.GradientConfig(f_wrapped, x_nominal, ForwardDiff.Chunk(x_nominal))
grad_wrapped! = function (g, θ)
    return ForwardDiff.gradient!(g, f_wrapped, θ, cfg)
end

# --- Timing the Gradient Evaluation ---
println("\n--- Timing Gradient Evaluation ---")
g = similar(x_nominal)

println("1) Compiling gradient (first call)...")
grad_wrapped!(g, x_nominal)

println("2) Timing gradient (second call)...")
@time grad_wrapped!(g, x_nominal)

println("Gradient at nominal: ", g)

# ==========================================
# 8. MDC Generation (Commented out for now)
# ==========================================
base_cost = CostFunction(f_wrapped, grad_wrapped!)
pipeline = TransformChain(LogAbsTransform())
final_cost = TransformedCost(base_cost, pipeline)

x_nominal_transformed = MinimallyDisruptiveCurves.inverse(pipeline, x_nominal)

# hess0 = ForwardDiff.hessian(θ -> final_cost(θ), x_nominal_transformed)
hess0 = [8126.935847493474 1002.977452237557 16178.76380564795 -45799.219069723506 -3.2460116135278216 -5998.620860457665 37508.47028176218 -29645.33878063824; 1002.9774522378033 44635.23545526759 6165.391593029518 53596.57850828468 -12349.082017653092 4938.243056269554 -51421.72858725375 31102.93926696443; 16178.763805647985 6165.391593029638 33837.6112317122 -86953.80086370997 730.5804180278384 -11705.694279270898 70367.23445357487 -56484.78754776795; -45799.21906972034 53596.57850828547 -86953.8008637117 338228.3315671754 -18523.786005955193 41639.76745430245 -286952.97719238297 213965.22893823672; -3.2460116134283963 -12349.082017653218 730.5804180279196 -18523.786005954695 6743.514764698752 -2043.9146794590392 16996.988403292384 -10957.545070173781; -5998.620860457804 4938.243056269316 -11705.694279271023 41639.767454302324 -2043.9146794590106 5221.34961898477 -35017.79557760796 26456.216948044475; 37508.47028176288 -51421.728587254474 70367.23445357545 -286952.97719236993 16996.988403292005 -35017.79557760929 244490.889720824 -181056.5622686294; -29645.338780637594 31102.93926696402 -56484.78754776701 213965.22893823174 -10957.545070174243 26456.216948043766 -181056.56226863287 135605.80728729363]
vs, vals = sparse_eigenbasis(hess0, 3; λ = 0.01)

mdc_curves = Dict{Int, Any}()

for i in 1:1
    println("\n--- Running MDC for index i = $i ---")
    _mdc_sys = MDCProblem(
        final_cost,
        x_nominal_transformed,
        vs[i],                
        1.0;                  
        names = params_to_optimize .|> Symbol
    )

    stabiliser = mdc_momentum_readjustment(_mdc_sys; tol = 1.0e-3)
    # logger =  mdc_verbose_callbacks(_mdc_sys, 0:0.05:1; is_negative=false)
    my_pipeline = CallbackSet(stabiliser)

    @time curves_i = MDCSolve(_mdc_sys, span = MDCSpan(-1.0, 1.0); callback = my_pipeline, mode=:fast)
    mdc_curves[i] = curves_i
end

println("\nMDC Generation Complete.")


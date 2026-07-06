using ModelingToolkit, Plots
using OrdinaryDiffEq
using ForwardDiff
using PreallocationTools
using SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_values
using SciMLStructures: Tunable, canonicalize, replace
using SciMLBase, LinearAlgebra

# Include your STG builder
using MTKNeuralToolkit
using MTKNeuralToolkit.PrinzNeuron

# ==========================================
# 1. Setup MTK Network & Base ODE Problem
# ==========================================
println("Building STG network...")
net = build_stg()
sys = mtkcompile(net.sys)
tspan = (0.0, 10000.0)

# Enable sparsity for the stiff STG system
prob = ODEProblem(sys, [], tspan, jac=true, sparse=true)

# ==========================================
# 2. Define the Specific Tracking States (Observables)
# ==========================================
target_observables = [
    sys.AB.AB_ca_pool.Ca,
    sys.LP.LP_ca_pool.Ca,
    sys.PY.PY_ca_pool.Ca,
]

# ==========================================
# 3. Dynamic Parameter Identification
# ==========================================
# We select 3 specific parameters to optimize (keeps Hessian computation fast)
params_to_optimize = [sys.AB.na.g, sys.AB.kdr.g, sys.ABLP_glut.g_max]

println("Natively optimizing $(length(params_to_optimize)) parameters.")

# ==========================================
# 4. Generate Baseline Experimental Data ("Truth")
# ==========================================
timesteps = 0.0:10.0:10000.0
sol_nominal = solve(prob, Rosenbrock23(); saveat = timesteps)

# Clean matrix extraction using the idxs keyword
truth_data = sol_nominal(timesteps, idxs = target_observables)

# ==========================================
# 5. High-Performance Loss Function
# ==========================================
function loss_function(x, p_tuple)
    # Destructure context tuple
    odeprob, ts, truth, setter, diffcache, obs_symbols = p_tuple

    ps = parameter_values(odeprob)
    buffer = get_tmp(diffcache, x)

    # Block-copy baseline values
    copyto!(buffer, canonicalize(Tunable(), ps)[1])

    # Type-safe structural parameter container replacement
    ps_updated = replace(Tunable(), ps, buffer)

    # Mutate only our active dual/float optimization array
    setter(ps_updated, x)

    # Fast inferred problem recreation
    newprob = remake(odeprob; p = ps_updated)
    sol = solve(newprob, Rosenbrock23(); saveat = ts)

    if sol.retcode != SciMLBase.ReturnCode.Success
        return eltype(x)(Inf) # Strict type stability for dual-number propagation
    end

    # Extract states cleanly via targeted tracking symbols
    current_data = sol(ts, idxs = obs_symbols)

    # Allocation-free MSE over the exact matrix of specified states
    return sum(abs2, truth .- current_data) / length(truth)
end

# ==========================================
# 6. Build the Optimization Context
# ==========================================
setter = setp(prob, params_to_optimize)
getter = getp(prob, params_to_optimize)

raw_ps = parameter_values(prob)
tunable_vector_prototype = copy(canonicalize(Tunable(), raw_ps)[1])
diffcache = DiffCache(tunable_vector_prototype)

p_tuple = (prob, timesteps, truth_data, setter, diffcache, target_observables)

# ==========================================
# 7. Evaluation and Gradient Verification
# ==========================================
println("\n--- Running Scaled Loss Function Evaluation ---")

x_nominal = getter(prob)
loss_at_nominal = loss_function(x_nominal, p_tuple)
println("Loss at nominal: ", loss_at_nominal)

# A clean, global-safe closure for the value calculation
f_wrapped = θ -> loss_function(θ, p_tuple)

# Pre-allocate the ForwardDiff configuration
cfg = ForwardDiff.GradientConfig(f_wrapped, x_nominal, ForwardDiff.Chunk(x_nominal))

# In-place wrapper function
grad_wrapped! = function (g, θ)
    return ForwardDiff.gradient!(g, f_wrapped, θ, cfg)
end

# ==========================================
# 8. MDC Pipeline Integration
# ==========================================
using MinimallyDisruptiveCurves

base_cost = CostFunction(f_wrapped, grad_wrapped!)
pipeline = TransformChain(LogAbsTransform())
final_cost = TransformedCost(base_cost, pipeline)
x_nominal_transformed = MinimallyDisruptiveCurves.inverse(pipeline, x_nominal)

println("Computing Hessian...")
@time hess0 = ForwardDiff.hessian(θ -> final_cost(θ), x_nominal_transformed)
vs, vals = sparse_eigenbasis(hess0, 3; λ = 0.01) # 3 directions for 3 params

# 1. Initialize an empty dictionary to store the results
mdc_curves = Dict{Int, Any}()

# 2. Loop through the desired indices
for i in 1:1
    println("--- Running MDC for index i = $i ---")

    # Create the system dynamically using the i-th direction
    _mdc_sys = MDCProblem(
        final_cost,
        x_nominal_transformed,
        vs[i],                # Replaced e_dirs(i) directly with vs[i]
        500.0;                  # Hamiltonian / momentum (H)
        names = params_to_optimize .|> Symbol
    )

    # Set up the pipeline for this iteration
    stabiliser = mdc_momentum_readjustment(_mdc_sys; tol = 1.0e-3)
    logger = mdc_verbose_callbacks(_mdc_sys, timepoints)

    my_pipeline = CallbackSet(stabiliser, logger)

    # Solve and store the result
    @time curves_i = MDCSolve(_mdc_sys, span = MDCSpan(-1.0, 1.0); callback = my_pipeline)

    mdc_curves[i] = curves_i
end

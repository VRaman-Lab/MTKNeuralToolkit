using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization
using OptimizationOptimJL
using SciMLStructures: Tunable, canonicalize
using SymbolicIndexingInterface: parameter_values, state_values
using SciMLBase: NoInit
using LinearAlgebra

# 1. Setup MTK Model
@parameters α β γ δ
@variables x(t) y(t)
eqs = [D(x) ~ (α - β * y) * x, D(y) ~ (δ * x - γ) * y]
@mtkcompile odesys = System(eqs, t)

odeprob = ODEProblem(odesys, [x => 1.0, y => 1.0, α => 1.5, β => 1.0, γ => 3.0, δ => 1.0], (0.0, 10.0))
timesteps = 0.0:0.1:10.0
sol = solve(odeprob, Tsit5(); saveat = timesteps)
truth = Array(sol) .+ 0.01 .* randn(size(Array(sol)))

# Extract the flat Tunable parameter array (order is [α, β, γ, δ])
flat_p = copy(canonicalize(Tunable(), parameter_values(odeprob))[1])

# 2. Define Loss and Grad using Direct Forward Sensitivity
function loss_and_grad(x, p)
    odeprob = p[1]
    timesteps = p[2]
    truth = p[3]
    
    f = odeprob.f
    u0 = state_values(odeprob)
    
    # Build the Forward Sensitivity Problem directly
    sens_prob = SciMLSensitivity.ODEForwardSensitivityProblem(f, u0, odeprob.tspan, x)
    # Workaround for missing initialization_data field
    sens_sol = solve(sens_prob, Tsit5(); saveat = timesteps, initializealg = NoInit())
    
    u_vals, dp_vals = SciMLSensitivity.extract_local_sensitivities(sens_sol)
    data = reduce(hcat, u_vals) 
    
    L = sum((truth .- data) .^ 2) / length(truth)
    dL_du = -2 .* (truth .- data) ./ length(truth)
    
    G = zeros(length(x))
    for i in 1:length(x)
        sens_i = reduce(hcat, dp_vals[i])
        G[i] = sum(dL_du .* sens_i)
    end
    
    return L, G
end

function loss(x, p)
    return loss_and_grad(x, p)[1]
end

function grad!(G, x, p)
    G .= loss_and_grad(x, p)[2]
    return G
end

# 3. Setup Optimization Problem
optfn = OptimizationFunction(loss, Optimization.AutoForwardDiff(); grad = grad!)
optprob = OptimizationProblem(optfn, flat_p, (odeprob, timesteps, truth), lb = 0.1ones(4), ub = 3ones(4))

println("Running Case 1 (Forward Sensitivity)...")
sol_opt = solve(optprob, BFGS())
println("Optimized parameters: ", sol_opt.u)

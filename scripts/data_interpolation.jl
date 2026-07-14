using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using DataInterpolations
using OrdinaryDiffEq
using Plots

# ==========================================
# 1. Generate Pseudo-Data
# ==========================================
dt = 0.1
t_data = collect(0.0:dt:10.0)
V_data = sin.(2 * pi * t_data / 5.0) .+ 0.1 .* randn.() 

# ==========================================
# 2. Create the Interpolation Object
# ==========================================
itp = LinearInterpolation(V_data, t_data)

# ==========================================
# 3. Create a Simple System that Consumes the Data
# ==========================================
@variables x(t) = 0.0
@parameters tau = 1.0

# We can call itp(t) directly! DataInterpolations handles the symbolic dispatch.
eqs = [
    D(x) ~ (itp(t) - x) / tau
]

@named tracking_sys = System(eqs, t, [x], [tau])

# ==========================================
# 4. Compile and Solve
# ==========================================
println("Compiling interpolation system...")
sys = mtkcompile(tracking_sys)

u0 = [sys.x => 0.0]
p = [sys.tau => 1.0]
prob = ODEProblem(sys, u0, (0.0, 10.0), p)

println("Solving...")
sol = solve(prob, Tsit5())

# ==========================================
# 5. Plot the Results
# ==========================================
p_plot = plot(sol, idxs=[sys.x], label="Model State (x)", linewidth=2)
plot!(p_plot, sol.t, itp.(sol.t), label="Interpolated Data", linestyle=:dash, color=:red)

title!("Driving an ODE with DataInterpolations")
xlabel!("Time (s)")
display(p_plot)

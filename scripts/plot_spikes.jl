using Plots
using Statistics: mean

dt = 0.1
τ = 1.0
α = dt / τ
tsteps = 0.0:dt:200.0
n = length(tsteps)

# Create a fake voltage trace with one spike at t=25ms
function fake_spike(tsteps, spike_time)
    v = fill(-65.0, length(tsteps))
    for (i, t) in enumerate(tsteps)
        if spike_time - 2.0 < t < spike_time
            v[i] = -65.0 + 10.0 * (t - (spike_time - 2.0)) / 2  # rise
        elseif spike_time <= t < spike_time + 0.5
            v[i] = -55.0  # peak
        elseif spike_time + 0.5 <= t < spike_time + 1.0

            v[i] = -70.0  # reset
        elseif spike_time + 1.0 <= t < spike_time + 10.0
            v[i] = -70.0 + 5.0 * (t - (spike_time + 1.0)) / 9.0  # recovery
        end
    end
    return v
end

function smooth_fwd_bwd(trace, α)
    n = length(trace)
    fwd = similar(trace)
    bwd = similar(trace)
    fwd[1] = trace[1]
    for k in 2:n
        fwd[k] = α * trace[k] + (1 - α) * fwd[k-1]
    end
    bwd[n] = trace[n]
    for k in (n-1):-1:1
        bwd[k] = α * trace[k] + (1 - α) * bwd[k+1]
    end
    return (fwd .+ bwd) ./ 2
end

# Two traces: spike at t=25 vs spike at t=26 (1ms shift)
trace1 = fake_spike(tsteps, 25.0)
trace2 = fake_spike(tsteps, 26.0)

smooth1 = smooth_fwd_bwd(trace1, α)
smooth2 = smooth_fwd_bwd(trace2, α)

# Compute pointwise squared error
raw_error = (trace1 .- trace2).^2
smooth_error = (smooth1 .- smooth2).^2

# ── Plot 1: Raw traces overlaid ───────────────────────
p1 = plot(collect(tsteps), trace1, linewidth=2, color=:black, 
          label="Spike at t=25ms")
plot!(p1, collect(tsteps), trace2, linewidth=2, color=:red, 
      label="Spike at t=26ms")
xlims!(p1, 15, 45)
ylabel!(p1, "Voltage (mV)")
title!(p1, "Raw Voltage Traces")

# ── Plot 2: Smoothed traces overlaid ──────────────────
p2 = plot(collect(tsteps), smooth1, linewidth=2, color=:black, 
          label="Spike at t=25ms")
plot!(p2, collect(tsteps), smooth2, linewidth=2, color=:red, 
      label="Spike at t=26ms")
xlims!(p2, 15, 45)
ylabel!(p2, "Voltage (mV)")
title!(p2, "Smoothed Voltage Traces (τ=1ms)")

# ── Plot 3: Squared error comparison ──────────────────
p3 = plot(collect(tsteps), raw_error, linewidth=2, color=:blue, 
          label="Raw MSE")
plot!(p3, collect(tsteps), smooth_error, linewidth=2, color=:green, 
      label="Smoothed MSE")
xlims!(p3, 15, 45)
ylabel!(p3, "Squared Error")
xlabel!(p3, "Time (ms)")
title!(p3, "Pointwise Squared Error")

plot(p1, p2, p3, layout=(3, 1), size=(800, 500))
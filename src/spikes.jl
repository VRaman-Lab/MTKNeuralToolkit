function detect_spikes(sol, var; mode::Symbol=:fixed,
                        threshold::Union{Float64,Nothing}=nothing,
                        min_prominence::Union{Float64,Nothing}=nothing,
                        refractory_period::Float64=1.0,
                        step_size::Float64=1.0,
                        interpolate::Bool=true)
    t, V = sol.t, sol[var]

    if mode == :fixed
        if threshold === nothing
            error("mode=:fixed requires `threshold`")
        end
        return _detect_fixed(t, V, threshold, refractory_period, step_size, interpolate)
    elseif mode == :prominence
        min_prominence === nothing && error("mode=:prominence requires `min_prominence`")
        return _detect_prominence(t, V, min_prominence, refractory_period)
    else
        error("unknown mode $mode")
    end
end

function _detect_fixed(t, V::AbstractVector{<:AbstractFloat}, threshold, refractory_period, step_size, interpolate)
    step_size = round(Int, step_size)
    spikes = Float64[]
    refractory_until = -Inf
    for i in 2:step_size:length(t)
        if V[i-1]<threshold<=V[i] && t[i] > refractory_until
            t_spike = t[i]
            if interpolate
                frac = (threshold - V[i-1]) / (V[i] - V[i-1])
                t_spike = t[i-1] + frac * (t[i] - t[i-1])
            end
            push!(spikes, t_spike)
            refractory_until = t_spike + refractory_period
        end
    end
    return spikes
end

function _detect_fixed(t, V::AbstractVector{<:AbstractVector}, threshold, refractory_period, step_size, interpolate)
    Vmat = stack(V; dims=1)
    n_neurons = size(Vmat, 2)
    spikes = Vector{Vector{Float64}}(undef, n_neurons)
    @views for j in 1:n_neurons
        spikes[j] = _detect_fixed(t, Vmat[:, j], threshold, refractory_period, step_size, interpolate)
    end
    return spikes
end

function _detect_prominence(t, V::AbstractVector{<:AbstractFloat}, min_prominence, refractory_period)

    V_range = maximum(V) - minimum(V)
    n = length(V)
    spikes = Float64[]
    refractory_until = -Inf

    for i in 2:n-1
        if V[i] > V[i-1] && V[i] > V[i+1] && t[i] > refractory_until            #Claude tells me scipy.signal.findpeaks does this better by sorting peaks by size
            peak = V[i]                                                         #And iterating to find and map troughs in this order to avoid duplicate walks;
                                                                                #This messes with refractory periods and makes things too complex -> error-prone.
            left_min, j = peak, i - 1                                           #What do you think?
            while j >= 1 && V[j] <= peak
                left_min = min(left_min, V[j]); j -= 1
            end

            right_min, k = peak, i + 1
            while k <= n && V[k] <= peak
                right_min = min(right_min, V[k]); k += 1
            end

            prominence_frac = (peak - max(left_min, right_min)) / V_range
            if prominence_frac >= min_prominence
                push!(spikes, t[i])
                refractory_until = t[i] + refractory_period
            end
        end
    end
    return spikes
end

function _detect_prominence(t, V::AbstractVector{<:AbstractVector}, min_prominence, refractory_period)
    n_neurons = length(V[1])
    V_neuron_major = [getindex.(V, j) for j in 1:n_neurons]
    spikes = [_detect_prominence(t, v, min_prominence, refractory_period) for v in V_neuron_major]
    return spikes
end
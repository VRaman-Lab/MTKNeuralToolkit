using ChainRulesCore

@mtkmodel BasicSoma begin
    @parameters begin
        C, [description = "Capacitance"]
    end
    @variables begin
        V(t) = -65.0, [description = "membrane voltage"]
    end
    @components begin
        oneport = OnePort()
        I = RealInput()
        ground = Ground()
    end
    @equations begin
        D(oneport.v) ~ (oneport.i + I.u) / C
        connect(ground.g, oneport.n)
        V ~ oneport.v
    end
end

"
Leaky Integrate-And-Fire soma where resetting dynamics are used 
This solves the Mass Matrix problem 
"

reset_function(V_reset, V_th, v) = v - (V_th - V_reset)  
@register_symbolic reset_function(V_reset, V_th, v)

σ(V_th, v; k=10.0) = 1 / (1 + exp(-k*(v-V_th)))
@register_symbolic σ(V_th, v)
σ′(V_th, v; k=10.0) = k * σ(V_th, v; k=k) * (1 - σ(V_th, v; k=k))
@register_symbolic σ′(V_th, v)


@mtkmodel LIFSoma begin
    @parameters begin
        C, [description = "Capacitance"]
        R
        V_reset = -70
        V_th = -55
        a = 1.0
    end
    @variables begin
        V(t) = -65, [description = "membrane voltage"]
        Spike_count(t) = 0
    end
    @components begin
        oneport = OnePort()
        I = RealInput()
        ground = Ground()
    end
    @equations begin
        D(oneport.v) ~ (oneport.i + I.u)/ C
        connect(ground.g, oneport.n)
        V ~ oneport.v
        D(Spike_count) ~ 0
    end
    @continuous_events begin
        [oneport.v ~ V_th] => (affect = [oneport.v ~ reset_function(V_reset, V_th, Pre(oneport.v)), Spike_count ~ Pre(Spike_count) + 1]) 
    end
end 

"""
A battery: generates a constant potential difference across its terminals
"""
@mtkmodel fixed_reversal begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        E
    end
    @equations begin
        v ~ E
    end
end


FixedReversal(;name = :reversal, kwargs...) = fixed_reversal(;name, kwargs...)


 

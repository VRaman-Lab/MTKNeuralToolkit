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

@mtkmodel LIFSoma begin
    @parameters begin
        C, [description = "Capacitance"]
        R
        V_reset = -70
        V_th = -55
        tau = 1
    end
    @variables begin
        V(t) = -65, [description = "membrane voltage"]
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
    end
    @continuous_events begin
        [oneport.v ~ V_th] => (affect = [oneport.v ~ Pre(V_reset)])
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


 

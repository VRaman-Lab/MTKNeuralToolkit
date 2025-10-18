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

function ModularSoma(;name=:conductance, continuous_events = nothing, C = 1.0, kwargs...)

    @named oneport = OnePort()
    @named I = RealInput()
    @named ground = Ground()
    
    @parameters begin
        C = C
    end

    @variables begin
        V(t) = -65.0
    end

    D = Differential(t)
    sys_eqs = [
        D(oneport.v) ~ (oneport.i + I.u) / C
        connect(ground.g, oneport.n)
        V ~ oneport.v
    ]

    if !isnothing(continuous_events)
        cuntenv = continuous_events(oneport.v)
    else
        cuntenv = []
    end

    sys = ODESystem(sys_eqs, t, name=name, [V], [C],
                    systems=[oneport, I, ground], continuous_events=cuntenv)
    return sys    
end

"""
A battery: generates a constant potential difference across its terminals
"""
@mtkmodel fixed_reversal begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        E = 0.0
    end
    @equations begin
        v ~ E
    end
end


FixedReversal(;name = :reversal, kwargs...) = fixed_reversal(;name, kwargs...)


 

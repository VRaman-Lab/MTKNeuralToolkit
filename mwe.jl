@mtkmodel nagates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E
    end
    @variables begin
        m_gate(t)=0.0, [description = "m gate"]
        h_gate(t)=1.0, [description = "h gate"]
        αₘ(t), [description = "opening"]
        αₕ(t), [description = "opening"]
        βₘ(t), [description = "closing"]
        βₕ(t), [description = "closing"]
    end
    @equations begin
        αₘ ~ 0.182(v+E+35)/(1. −exp(−(v+E+35.)/ 9.))
        βₘ ~ -0.124(v+E+35)/(1. −exp((v+E+35.)/ 9.))
        αₕ ~ 0.25*exp(−(v+E+90.)/12.) 
        βₕ ~ 0.25*(exp((v+E+62.)/6.))/exp((v+E+90.)/12.) 
        D(m_gate) ~  αₘ * (1 - m_gate) - βₘ * m_gate
        D(h_gate) ~ αₕ* (1 - h_gate) - βₕ * h_gate
        i ~ g * m_gate^3*h_gate * v 
    end
end

@component BasicSoma(;name)
    params = @parameters begin
        C, [description = "Capacitance"]
    end
    vars = @variables begin
        V(t) = -65.0, [description = "membrane voltage"]
    end
    comps = @components begin
        oneport = OnePort()
        I = RealInput()
        ground = Ground()
    end
    eqs = [
        D(oneport.v) ~ (oneport.i + I.u) / C
        connect(ground.g, oneport.n)
        V ~ oneport.v]
    System(eqs, t, vars, params ; name, components)
end

"""
A battery: generates a constant potential difference across its terminals
"""
@component fixed_reversal(;name)
    @extend v, i = oneport = OnePort()
    params = @parameters begin
        E
    end
    eqs = [
        v ~ E]
System(eqs, t, [], params; name)
end


FixedReversal(;name = :reversal, kwargs...) = fixed_reversal(;name, kwargs...)


function build_channel(conductance, reversal;name)
    if conductance.p === nothing
        return build_channel_explicit(conductance;name, reversal=reversal)
    end
    @named p = Pin()
    @named n = Pin()
    connections = [
        connect(conductance.p, reversal.n),
        connect(conductance.n, n),
        connect(reversal.p, p)
    ]
    return compose(ODESystem(connections, t; name), [p,n,conductance,reversal])
end

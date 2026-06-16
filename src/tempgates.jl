"""
nagates Component: Pure Sodium channel gating resistance. 
"""
@component function nagates(; name, g = 120.0) 
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters g = g
    vars = @variables begin
        m_gate(t)=0.0
        h_gate(t)=1.0
        αₘ(t)
        αₕ(t)
        βₘ(t)
        βₕ(t)
    end
    eqs = [
        αₘ ~ 0.182 * (v + 35.0) / (1.0 - exp(-(v + 35.0) / 9.0))
        βₘ ~ -0.124 * (v + 35.0) / (1.0 - exp((v + 35.0) / 9.0))
        αₕ ~ 0.25 * exp(-(v + 90.0) / 12.0) 
        βₕ ~ 0.25 * (exp((v + 62.0) / 6.0)) / exp(-(v + 90.0) / 12.0) 
        D(m_gate) ~ αₘ * (1 - m_gate) - βₘ * m_gate
        D(h_gate) ~ αₕ * (1 - h_gate) - βₕ * h_gate
        i ~ g * m_gate^3 * h_gate * v 
    ]
    extend(System(eqs, t, vars, [g]; name), oneport)
end

"""
kgates Component: Pure Potassium channel gating resistance.
"""
@component function kgates(; name, g = 36.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters g = g
    vars = @variables begin
        n_gate(t) = 0.0
        αₙ(t)
        βₙ(t)
    end
    eqs = [
        αₙ ~ 0.02 * (v - 25.0) / (1.0 - exp(-(v - 25.0) / 9.0))
        βₙ ~ -0.002 * (v - 25.0) / (1.0 - exp((v - 25.0) / 9.0))
        D(n_gate) ~ αₙ * (1 - n_gate) - βₙ * n_gate
        i ~ v * n_gate^4 * g
    ]
    extend(System(eqs, t, vars, [g]; name), oneport)
end

"""
lgates Component: Pure passive Leak channel resistance.
"""
@component function lgates(; name, g = 0.3)
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters g = g
    eqs = [i ~ v * g]
    extend(System(eqs, t, [], [g]; name), oneport)
end

export nagates,lgates,kgates

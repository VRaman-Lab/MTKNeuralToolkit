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

@mtkmodel nagates_opt begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E
    end
    @variables begin
        m_gate(t)=0.0, [description = "m gate"]
        h_gate(t)=1.0, [description = "h gate"]
    end
    @equations begin
        D(m_gate) ~  (0.182(v+E+35)/(1. −exp(−(v+E+35.)/ 9.))) * (1 - m_gate) - (-0.124(v+E+35)/(1. −exp((v+E+35.)/ 9.))) * m_gate
        D(h_gate) ~ (0.25*exp(−(v+E+90.)/12.) )* (1 - h_gate) - (0.25*(exp((v+E+62.)/6.))/exp((v+E+90.)/12.)) * h_gate
        i ~ g * m_gate^3*h_gate * v 
    end
end

"""
Leak Channel is just a conductance
"""
@mtkmodel lgates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g = 10.0, [description = "Conductance"]
        E
    end
    @equations begin
        i ~ v*g
    end
end

"""
Just a conductance
"""
@mtkmodel kgates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g = 35.0, [description = "Conductance"]
        E,[description = "Reversal"]
    end
    @variables begin
        n_gate(t) = 0.0, [description = "Potassium gate"]
        αₙ(t), [description = "opening"]
        βₙ(t), [description = "closing"]
    end
    @equations begin
        D(n_gate) ~ (0.02(v +E − 25.) / (1. − exp(− (v +E − 25.) /  9.))) * (1 - n_gate) - (-0.002(v+E − 25.) / (1. − exp( (v+E − 25.) /  9.))) * n_gate
        i ~ v*n_gate^4*g
    end
end
#=
@mtkmodel kgates_new begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g = 35.0, [description = "Conductance"]
        E,[description = "Reversal"]
    end
    @variables begin
        n_gate(t) = 0.0, [description = "Potassium gate"]
        αₙ(t) ~ 0.02(v +E − 25.) / (1. − exp(− (v +E − 25.) /  9.))
        βₙ(t) ~  -0.002(v+E − 25.) / (1. − exp( (v+E − 25.) /  9.))
    end
    @equations begin
        D(n_gate) ~ αₙ * (1 - n_gate) - βₙ * n_gate
        i ~ v*n_gate^4*g
    end
end
=#
NaGates(  ;name=:conductance , kwargs...) = nagates( ;name , kwargs...)
KGates( ;name=:conductance , kwargs...) = kgates( ;name, kwargs...)
LGates( ;name=:conductance , kwargs...) = lgates(;name , kwargs...)

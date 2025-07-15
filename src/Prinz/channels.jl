#Prinz channels
@mtkmodel nagates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E
    end
    @variables begin
        m(t)=0.0, [description = "m gate"]
        h(t)=1.0, [description = "h gate"]
        m∞(t), [description = "steady state m opening"]
        h∞(t), [description = "steady state m opening"]
        τm(t), [description = "m gate time constant"]
        τh(t), [description = "h gate time constant"]
    end
    @equations begin
        m∞ ~ 1.0 / (1.0 + exp((v+E + 25.5) / -5.29))
        h∞ ~ 1.0 / (1.0 + exp((v+E + 48.9) / 5.18))
        τm ~ 2.64 - 2.52 / (1 + exp((v+E + 120.0) / -25.0))
        τh ~ (1.34 / (1.0 + exp((v+E + 62.9) / -10.0))) * (1.5 + 1.0 / (1.0 + exp((v+E + 34.9) / 3.6)))
        D(m) ~  (1/τm)*(m∞ - m) 
        D(h) ~ (1/τh)*(h∞ - h)
        i ~ g * m^3*h * v 
    end
end


@mtkmodel casgates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
    end
    @variables begin
        Ca(t), [description = "Calcium concentration"]
        m(t)=0.0, [description = "m gate"]
        h(t)=1.0, [description = "h gate"]
        m∞(t), [description = "steady state m opening"]
        h∞(t), [description = "steady state m opening"]
        τm(t), [description = "m gate time constant"]
        τh(t), [description = "h gate time constant"]
        E(t), [description = "reversal potential"]
    end
    @components begin
        ca = IonicPort()
        ICa = RealInput()
    end
    @equations begin
        m∞ ~ 1.0 / (1.0 + exp((v+E + 33.0) / -8.1))
        h∞ ~ 1.0 / (1.0 + exp((v+E + 60.0) / 6.2))
        τm ~ 2.8 + 14.0 / (exp((v+E + 27.0) / 10.0) + exp((v+E + 70.0) / -13.0))
        τh ~ 120.0 + 300.0 / (exp((v+E + 55.0) / 9.0) + exp((v+E + 65.0) / -16.0))
        ca.i ~ i
        Ca ~ ca.q
        ICa.u ~ i
        D(m) ~  (1/τm)*(m∞ - m) 
        D(h) ~ (1/τh)*(h∞ - h)
        E ~ (500.0) * (8.6174e-5) * (283.15) * log(max((3000.0 / Ca), 0.001))

        #i ~ g * m^3*h * v 
        i ~ g * m^3*h * (v)
    end
end

@mtkmodel catgates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
    end
    @variables begin
        Ca(t), [description = "Calcium concentration"]
        m(t)=0.0, [description = "m gate"]
        h(t)=1.0, [description = "h gate"]
        m∞(t), [description = "steady state m opening"]
        h∞(t), [description = "steady state m opening"]
        τm(t), [description = "m gate time constant"]
        τh(t), [description = "h gate time constant"]
        E(t), [description = "reversal potential"]
    end
    @components begin
        ca = IonicPort()
        ICa = RealInput()
    end
    @equations begin
        m∞ ~ 1.0 / (1.0 + exp((v+E + 27.1) / -7.2))
        h∞ ~ 1.0 / (1.0 + exp((v+E + 32.1) / 5.5))
        τm ~ 43.4 - 42.6 / (1.0 + exp((v+E +68.1) / -20.5))
        τh ~ 210. - 179.6 / (1.0 + exp((v+E + 55.0) / 16.9))
        ca.i ~ i
        Ca ~ ca.q
        ICa.u ~ i
        D(m) ~  (1/τm)*(m∞ - m) 
        D(h) ~ (1/τh)*(h∞ - h)
        E ~ (500.0) * (8.6174e-5) * (283.15) * log(max((3000.0 / Ca), 0.001))

        #i ~ g * m^3*h * v 
        i ~ g * m^3*h * (v)
    end
end

@mtkmodel kcagates begin
    @extend v, i = oneport = OnePort()          #For listening to 
    @parameters begin
        g, [description = "Conductance"]
        E, [description = "Reversal"]
    end
    @variables begin
        Ca(t), [description = "Calcium concentration"]
        m(t)=0.0, [description = "m gate"]
        h(t)=1.0, [description = "h gate"]
        m∞(t), [description = "steady state m opening"]
        τm(t), [description = "m gate time constant"]
    end
    @components begin
        ca = IonicTerminal()
        ICa = RealInput()
    end
    @equations begin
        m∞ ~ (Ca / (Ca + 3.0)) / (1.0 + exp((v+E + 28.3) / -12.6));
        τm ~ 180.6 - 150.2 / (1.0 + exp((v+E + 46.0) / -22.7));
        D(m) ~  (1/τm)*(m∞ - m) 
        i ~ g * m^4 * v 
        Ca ~ ca.q
        ICa.u ~ i
    end
end

@mtkmodel kgates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E
    end
    @variables begin
        m(t)=0.0, [description = "m gate"]
        h(t)=1.0, [description = "h gate"]
        m∞(t), [description = "steady state m opening"]
        h∞(t), [description = "steady state m opening"]
        τm(t), [description = "m gate time constant"]
        τh(t), [description = "h gate time constant"]
    end
    @equations begin
        m∞ ~ 1.0 / (1.0 + exp((v+E + 27.2) / -8.7))
        h∞ ~ 1.0 / (1.0 + exp((v+E + 56.9) / 4.9))
        τm ~ 23.2 - 20.8 / (1.0 + exp((v+E + 32.9) / -15.2))
        τh ~ 77.2 - 58.4 / (1.0 + exp((v+E + 38.9) / -26.5))
        D(m) ~  (1/τm)*(m∞ - m) 
        D(h) ~ (1/τh)*(h∞ - h)
        i ~ g * m^3*h * v 
    end
end

@mtkmodel drkgates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E
    end
    @variables begin
        m(t)=0.0, [description = "m gate"]
        m∞(t), [description = "steady state m opening"]
        τm(t), [description = "m gate time constant"]
    end
    @equations begin
        m∞ ~ 1.0 / (1.0 + exp((v+E + 12.3) / -11.8))
        τm ~ 14.4 - 12.8 / (1.0 + exp((v+E + 28.3) / -19.2))
        D(m) ~  (1/τm)*(m∞ - m) 
        i ~ g * m^4 * v 
    end
end

@mtkmodel hgates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E
    end
    @variables begin
        m(t)=0.0, [description = "m gate"]
        m∞(t), [description = "steady state m opening"]
        τm(t), [description = "m gate time constant"]
    end
    @equations begin
        m∞ ~ 1.0 / (1.0 + exp((v+E + 75.0) / 5.5))
        τm ~ ( 2 / exp((v + 169.7) / (-11.6) + exp((v - 26.7) / 14.3)))
        D(m) ~  (1/τm)*(m∞ - m) 
        i ~ g * m * v 
    end
end

@mtkmodel leakgates begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        g, [description = "Conductance"]
        E
    end
    @equations begin
        i ~ g * (E - v)
    end
end


NaGates(  ;name=:conductance , kwargs...) = nagates( ;name , kwargs...)
KCaGates( ;name=:conductance , kwargs...) = kcagates( ;name, kwargs...)
CaSGates( ;name=:conductance , kwargs...) = casgates(;name , kwargs...)
CalciumReversal( ;name=:reversal , kwargs...)  = calciumreversal(;name , kwargs...)
CaTGates( ;name=:conductance , kwargs...) = catgates(;name , kwargs...)
KGates( ;name=:conductance , kwargs...) = kgates(;name, kwargs...)
DRKGates( ;name=:conductance , kwargs...) = drkgates(;name , kwargs...)
HGates( ;name=:conductance , kwargs...) = hgates(;name , kwargs...)
LeakGates( ;name=:conductance , kwargs...) = leakgates(;name , kwargs...)
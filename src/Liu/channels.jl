# @mtkmodel LiuCalciumDynamics begin
#     @structural_parameters begin
#         flux_multiplier =  0.939488
#         Ca∞ = 0.5
#     end
#     @variables begin
#         Ca(t) = 0.5, [description = "calcium concentration"]
#         E(t), [description = "calcium reversal"]
#     end
#     @parameters begin
#         C, [description = "neuron capacitance"]
#         τ = 10.0, [description = "calcium time constant"] 
#     end
#     @equations begin
#         D(Ca) ~ (1 / τCa) * (-Ca + Ca∞ + (flux_multiplier * currents / C))
#     end
# end

@mtkmodel calciumreversal begin
    @extend v, i = oneport = OnePort()
    @variables begin
        V(t), [description = "dynamic voltage"]
        Ca(t)
    end
    @components begin
        ca = IonicPort()
    end
    @equations begin
        V ~ (500.0) * (8.6174e-5) * (283.15) * log(max((3000.0 / Ca),0.001))
        Ca ~ ca.q
        V ~ v
    end
end


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
        τm ~ 1.32 - 1.26 / (1 + exp((v+E + 120.0) / -25.0))
        τh ~ (0.67 / (1.0 + exp((v+E + 62.9) / -10.0))) * (1.5 + 1.0 / (1.0 + exp((v+E + 34.9) / 3.6)))
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
        m(t)=0.0, [description = "m gate"]
        h(t)=1.0, [description = "h gate"]
        m∞(t), [description = "steady state m opening"]
        h∞(t), [description = "steady state m opening"]
        τm(t), [description = "m gate time constant"]
        τh(t), [description = "h gate time constant"]
        E(t), [description = "reversal potential"]
    end
    @equations begin
        m∞ ~ 1.0 / (1.0 + exp((v+E + 33.0) / -8.1))
        h∞ ~ 1.0 / (1.0 + exp((v+E + 60.0) / 6.2))
        τm ~ 1.4 + 7.0 / (exp((v+E + 27.0) / 10.0) + exp((v+E + 70.0) / -13.0))
        τh ~ 60.0 + 150.0 / (exp((v+E + 55.0) / 9.0) + exp((v+E + 65.0) / -16.0))
        D(m) ~  (1/τm)*(m∞ - m) 
        D(h) ~ (1/τh)*(h∞ - h)
        v ~ E # TODO CHECK CONSISTENCY!!!
        i ~ g * m^3*h * v 
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
        τm ~ 90.3 - 75.1 / (1.0 + exp((v+E + 46.0) / -22.7));
        D(m) ~  (1/τm)*(m∞ - m) 
        i ~ g * m^4 * v 
        ca.i ~ i
        Ca ~ ca.q
        ICa.u ~ i
    end
end

# connect(Kca.a.p, soma.ca.n )
# connect(Kca.)


NaGates(  ;name=:conductance , kwargs...) = nagates( ;name , kwargs...)
KCaGates( ;name=:conductance , kwargs...) = kcagates( ;name, kwargs...)
CaSGates( ;name=:conductance , kwargs...) = casgates(;name , kwargs...)
CalciumReversal( ;name=:reversal , kwargs...)  = calciumreversal(;name , kwargs...)

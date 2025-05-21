@connector IonicPin begin
    q(t)                  # Concentration at the pin [V]
    i(t), [connect = Flow]    # Current flowing into the pin [A]
end

@mtkmodel IonicPort begin
    @components begin
        p = IonicPin()
        n = IonicPin()
    end
    @variables begin
        q(t)
        i(t)
    end
    @equations begin
        q ~ p.q - n.q
        0 ~ p.i + n.i
        i ~ p.i
    end
end



@mtkmodel IonicTerminal begin
    @components begin
        p = IonicPin()
        n = IonicPin()
    end
    @variables begin
        q(t)
        i(t)
    end
    @equations begin
        q ~ p.q 
        n.q ~ 0
        n.i ~ 0
        i ~ p.i
    end
end



@mtkmodel IonicGround begin
    @components begin
        g = IonicPin()
    end
    @equations begin
        g.q ~ 0
    end
end





# @mtkmodel CaSGates begin
#     @extend v, i = oneport = OnePort()
#     @parameters begin
#         g, [description = "Conductance"]
#     end
#     @variables begin
#         m(t)=0.0, [description = "m gate"]
#         h(t)=1.0, [description = "h gate"]
#         m∞(t), [description = "steady state m opening"]
#         h∞(t), [description = "steady state m opening"]
#         τm(t), [description = "m gate time constant"]
#         τh(t), [description = "h gate time constant"]
#         E(t), [description = "reversal potential"]
#     end
#     @equations begin
#         m∞ ~ 1.0 / (1.0 + exp((v+E + 33.0) / -8.1))
#         h∞ ~ 1.0 / (1.0 + exp((v+E + 60.0) / 6.2))
#         τm ~ 1.4 + 7.0 / (exp((v+E + 27.0) / 10.0) + exp((v+E + 70.0) / -13.0))
#         τh ~ 60.0 + 150.0 / (exp((v+E + 55.0) / 9.0) + exp((v+E + 65.0) / -16.0))
#         D(m) ~  (1/τm)*(m∞ - m) 
#         D(h) ~ (1/τh)*(h∞ - h)
#         i ~ g * m^3*h * v 
#     end
# end

# @mtkmodel KCaGates begin
#     @extend v, i = oneport = OnePort()
#     @parameters begin
#         g, [description = "Conductance"]
#         E, [description = "Reversal"]
#     end
#     @variables begin
#         Ca(t), [description = "Calcium concentration"]
#         m(t)=0.0, [description = "m gate"]
#         h(t)=1.0, [description = "h gate"]
#         m∞(t), [description = "steady state m opening"]
#         τm(t), [description = "m gate time constant"]
#     end
#     @equations begin
#         m∞ ~ (Ca / (Ca + 3.0)) / (1.0 + exp((v+E + 28.3) / -12.6));
#         τm ~ 90.3 - 75.1 / (1.0 + exp((v+E + 46.0) / -22.7));
#         D(m) ~  (1/τm)*(m∞ - m) 
#         i ~ g * m^4 * v 
#     end
# end



# @mtkmodel LiuDynamicVoltage begin
#     @extend v, i = oneport = OnePort()
#     @variables begin
#         V(t), [description = "dynamic voltage"]
#         Ca(t)
#     end
#     @equations begin
#         V ~ (500.0) * (8.6174e-5) * (283.15) * log(max((3000.0 / Ca),0.001))
#     end
# end

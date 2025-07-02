@mtkmodel CalciumSensitiveNeuron begin
    @parameters begin
        C, [description = "Capacitance"]
        flux_multiplier =  0.939488
        Ca∞ = 0.5
        τ = 10.0, [description = "calcium time constant"] 
    end
    @extend v, i = oneport = OnePort(; v)
    @variables begin
        Ca(t) = 0.5, [description = "calcium concentration"]
        V(t) = -65.0, [description = "membrane voltage"]
    end
    @components begin
        I = RealInput()
        ground = Ground()
        CaGround = IonicGround()
        ca = IonicPort()
    end
    @equations begin
        D(v) ~ (i + I.u) / C
        connect(ground.g, oneport.n)
        connect(CaGround.g, ca.n)
        V ~ v
        D(Ca) ~ (1 / τ) * (-Ca + Ca∞ + (flux_multiplier * ca.i / C))
        Ca ~ ca.q
    end
end


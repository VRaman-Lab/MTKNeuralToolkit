@mtkmodel CalciumSensitiveNeuron begin
    @parameters begin
        C, [description = "Capacitance"]
        flux_multiplier =  0.939488
        Ca∞ = 0.5
        τ = 200.0, [description = "calcium time constant"] 
    end
    @variables begin
        Ca(t) = 0.5, [description = "calcium concentration"]
        V(t) = -65.0, [description = "membrane voltage"]
    end
    @components begin
        oneport = OnePort()
        I = RealInput()
        ground = Ground()
        CaGround = IonicGround()
        ca = IonicPort()
    end
    @equations begin
        D(oneport.v) ~ (oneport.i + I.u) / C
        connect(ground.g, oneport.n)
        connect(CaGround.g, ca.n)
        V ~ oneport.v
        D(Ca) ~ (1 / τ) * (-Ca + Ca∞ + (flux_multiplier * ca.i / C))
        Ca ~ ca.q
    end
end
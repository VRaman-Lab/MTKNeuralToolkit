@component function BasicSoma(;name)
    params = @parameters begin
        C, [description = "Capacitance"]
    end
    vars = @variables begin
        V(t) = -65.0, [description = "membrane voltage"]
    end
    @named onePort = OnePort()
    @named I = RealInput()
    @named ground = Ground()

    # comps = [onePort, I, ground]

    eqs = [
        D(onePort.v) ~ (onePort.i + I.u) / C,
        connect(ground.g, onePort.n),
        V ~ onePort.v]
    System(eqs, t, vars, params ; name, onePort,I,ground)
end

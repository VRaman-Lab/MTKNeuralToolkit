"""@component function FOL(;name)
    params = params = @parameters begin
        τ = 3.0 # parameters
    end
    vars = vars = @variables begin
        x(t) = 0.0 # dependentvars =  variables
    end
    eqs = [
        D(x) ~ (1 - x) / τ
    ]
    System(eqs, t, vars, params; name)
end

@mtkcompile fol = FOL()"""


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


 

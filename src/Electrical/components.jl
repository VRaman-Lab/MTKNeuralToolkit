
"""
Represents a pure physical lipid bilayer membrane patch.
"""
@component function Capacitor(; name, C = 1.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters C = C
    vars = @variables V(t) = -65.0
    
    eqs = [
        D(v) ~ i / C
        V ~ v
    ]
    extend(System(eqs, t, vars, [C]; name), oneport)
end

"""
fixed_reversal Component: A pure constant voltage source (Nernst battery).
"""
@component function fixed_reversal(; name, E = 0.0)
    @named oneport = OnePort()
    @unpack v = oneport
    params = @parameters E = E
    eqs = [v ~ E]
    extend(System(eqs, t, [], [E]; name), oneport)
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


 

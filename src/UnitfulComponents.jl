"""
DEPRECATED

An attempt to redraw the basic components with DynamicQuantities.jl units.

Decided to kill. Makes everything error-prone and there are inherent deficiences such as requiring the same scaling of units.

Better option is to allow users the option of converting their parametesr into a choice of SI units and passing those into MTK

Then allowing conversion of sol into a desired choice of units.
    
"""


"""
Soma Component: Represents a pure physical lipid bilayer membrane patch.
"""
@connector function Pin(; name, v = nothing, i = nothing)
    vars = @variables begin
        v(t) = v, [unit = u"mV"]                  # Potential at the pin [V]
        i(t) = i, [connect = Flow, unit = u"nA"]    # Current flowing into the pin [A]
    end
    System(Equation[], t, vars, []; name)
end

@component function OnePort(; v = nothing, i = nothing, name)
    pars = @parameters begin
    end

    systems = @named begin
        p = Pin()
        n = Pin()
    end

    vars = @variables begin
        v(t) = v, [unit = u"mV"]
        i(t) = i, [unit = u"nA"]
    end

    equations = Equation[
        v ~ p.v - n.v,
        0 ~ p.i + n.i,
        i ~ p.i,
    ]

    return System(equations, t, vars, pars; name, systems)
end

@component function Ground(; name)
    pars = @parameters begin
    end

    systems = @named begin
        g = Pin()
    end

    vars = @variables begin
    end

    equations = Equation[
        g.v ~ 0,
    ]

    return System(equations, t, vars, pars; name, systems)
end



@connector function RealInput(;
        name, nin = 1, u_start = nothing, guess = nin > 1 ? zeros(nin) : 0.0
    )
    if u_start !== nothing
        Base.depwarn(
            "The keyword argument `u_start` is deprecated. Use `guess` instead.", :u_start
        )
        guess = u_start
    end
    if nin == 1
        @variables u(t) [
            input = true,
            description = "Inner variable in RealInput $name",
            unit = u"nA"
        ]
    else
        @variables u(t)[1:nin] [
            input = true,
            description = "Inner variable in RealInput $name",
            unit = u"nA"
        ]
    end
    System(Equation[], t, [u;], []; name = name, guesses = [u => guess])
end


@component function Capacitor(; name, C = 1.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    params = @parameters C = C, [unit = u"μF"]
    vars = @variables V(t) = -65.0, [unit = u"mV"]
    
    eqs = [
        D(v) ~ i / C
        V ~ v
    ]
    extend(System(eqs, t, vars, [C]; name), oneport)
end

@component function LIFCapacitor(; name, C = 10.0, V_th = -55.0, V_reset = -67.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    params = @parameters(
        C = C, [unit = u"μF"],
        V_th = V_th, [unit = u"mV"],
        V_reset = V_reset, [unit = u"mV"]
    )
    vars = @variables V(t) = -65.0, [unit=u"mV"]
    
    eqs = [
        D(v) ~ i / C
        V ~ v
    ]
    
    root_eqs = [v ~ V_th] 
    # Affect: The state after the callback becomes V_reset
    affect   = [v ~ V_reset] 
    
    # Combine via Pair mapping: conditions => affects
    events = root_eqs => affect
    
    base_sys = System(eqs, t, vars, [C, V_th, V_reset]; name, continuous_events = events)
    return extend(base_sys, oneport)
end


"""
CurrentSource Component: Converts a causal RealInput signal (u) 
into an acausal electrical current (i) injecting into a physical Node.
"""
@component function CurrentSource(; name)
    @named oneport = OnePort()
    @unpack i = oneport
    @named I = RealInput()
    
    eqs = [
        i ~ I.u 
    ]
    extend(System(eqs, t, [], []; name, systems = [I]), oneport)
end


"""
fixed_reversal Component: A pure constant voltage source (Nernst battery).
"""
@component function FixedReversal(; name, E = 0.0)
    @named oneport = OnePort()
    @unpack v = oneport
    params = @parameters E = E, [unit=u"mV"]
    eqs = [v ~ E]
    extend(System(eqs, t, [], [E]; name), oneport)
end

# @named soma = LIFCapacitor(C = 1.0)
# cc = build_neuron(soma, [])nd

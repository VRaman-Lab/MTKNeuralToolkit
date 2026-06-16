"""
Soma Component: Represents a pure physical lipid bilayer membrane patch.
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

# """
# CurrentSource Component: Converts a causal RealInput signal (u) 
# into an acausal electrical current (i) injecting into a physical Node.
# """
# @component function CurrentSource(; name)
#     @named oneport = OnePort()
#     @unpack i = oneport
#     @named I = RealInput()
    
#     eqs = [
#         i ~ -I.u 
#     ]
#     extend(System(eqs, t, [], []; name, systems = [I]), oneport)
# end


"""
CurrentSource Component: Converts a causal RealInput signal (u) 
into an acausal electrical current (i) injecting into a physical Node.
"""
@component function CurrentSource(; name)
    @named oneport = OnePort()
    @unpack i = oneport
    @named I = RealInput()
    
    eqs = [
        i ~ -I.u 
    ]
    extend(System(eqs, t, [], []; name, systems = [I]), oneport)
end


"""
fixed_reversal Component: A pure constant voltage source (Nernst battery).
"""
@component function FixedReversal(; name, E = 0.0)
    @named oneport = OnePort()
    @unpack v = oneport
    params = @parameters E = E
    eqs = [v ~ E]
    extend(System(eqs, t, [], [E]; name), oneport)
end


"""
LIFCapacitor Component: Capacitor that automatically resets its voltage when a threshold is crossed 
"""
@component function LIFCapacitor(; name, C = 10.0, V_th = -55.0, V_reset = -67.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    params = @parameters(
        C = C,
        V_th = V_th,
        V_reset = V_reset
    )
    vars = @variables V(t) = -65.0
    
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


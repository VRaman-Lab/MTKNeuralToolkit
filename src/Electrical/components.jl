using ChainRulesCore
using SciMLStructures
using SymbolicIndexingInterface

@mtkmodel BasicSoma begin
    @parameters begin
        C, [description = "Capacitance"]
    end
    @variables begin
        V(t) = -65.0, [description = "membrane voltage"]
    end
    @components begin
        oneport = OnePort()
        I = RealInput()
        ground = Ground()
    end
    @equations begin
        D(oneport.v) ~ (oneport.i + I.u) / C
        connect(ground.g, oneport.n)
        V ~ oneport.v
    end
end

"
Leaky Integrate-And-Fire soma where resetting dynamics are used 
This solves the Mass Matrix problem 
"
@mtkmodel LIFSoma begin
    @parameters begin
        C, [description = "Capacitance"]
        R
        V_reset = -70
        V_th = -55
        a = 1.0
    end
    @variables begin
        V(t) = -65, [description = "membrane voltage"]
        Spike_count(t) = 0
    end
    @components begin
        oneport = OnePort()
        I = RealInput()
        ground = Ground()
    end
    @equations begin
        D(oneport.v) ~ (oneport.i + I.u)/ C
        connect(ground.g, oneport.n)
        V ~ oneport.v
        D(Spike_count) ~ 0
    end
end 

"""
A battery: generates a constant potential difference across its terminals
"""
@mtkmodel fixed_reversal begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        E
    end
    @equations begin
        v ~ E
    end
end


FixedReversal(;name = :reversal, kwargs...) = fixed_reversal(;name, kwargs...)


 

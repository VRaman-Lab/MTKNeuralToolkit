using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D, @named, @parameters, @variables, @component, @unpack, Equation, System, extend
using Symbolics: SymbolicT
using DataInterpolations
using ..MTKNeuralToolkit: OnePort, VectorizedOnePort, Scalar, Vectorized, get_conductance

"""
    PEMObservationChannel(; name, itp, K_init=1.0, topology=Scalar())

An acausal channel that injects an observer current to drive the membrane voltage 
towards a target interpolated dataset. Used for the Prediction Error Method (PEM).

Equations:
    i ~ K * (itp(t) - v)

Where `itp` is a DataInterpolations object, `K` is the observer gain (optimized during PEM), 
and `v` is the membrane voltage of the OnePort.
"""
@component function PEMObservationChannel(; name, itp, K_init=1.0, topology=Scalar())
    if topology isa Scalar
        @named oneport = OnePort()
        @unpack v, i = oneport
        
        @parameters K = K_init
        params = SymbolicT[K]
        
        vars = SymbolicT[]
        
        # The observer correction current
        eqs = Equation[
            i ~ K * (v - itp(t))
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        
    else
        N = topology.N
        @named oneport = VectorizedOnePort(N=N)
        @unpack v, i = oneport
        
        if K_init isa AbstractArray
            @parameters K[1:N] = K_init
        else
            @parameters K = K_init
        end
        params = SymbolicT[K]
        
        vars = SymbolicT[]
        
        # Broadcast the interpolation over the vectorized states
        eqs = Equation[
            i ~ K .* (v .- itp(t))
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
    end
end

@component function PEMObservationChannel(; name, itps::AbstractVector, K_init=1.0, topology=Scalar())
    N = length(itps)
    
    if topology isa Scalar
        @named oneport = OnePort()
        @unpack v, i = oneport
        
        @parameters K = K_init
        params = SymbolicT[K]
        vars = SymbolicT[]
        
        # Explicitly use the first element of the array
        eqs = Equation[
            i ~ K * (v - itps[1](t))
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        
    else
        @named oneport = VectorizedOnePort(N=N)
        @unpack v, i = oneport
        
        if K_init isa AbstractArray
            @parameters K[1:N] = K_init
        else
            @parameters K = K_init
        end
        params = SymbolicT[K]
        vars = SymbolicT[]
        
        # Clean, explicit unrolling to guarantee MTK shape inference
        target_vec = SymbolicT[itps[j](t) for j in 1:N]
        
        eqs = Equation[
            i ~ K .* (v .- target_vec)
        ]
        
        return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
    end
end

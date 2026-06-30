module ContinuousSpikers
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, Scalar, Vectorized, OnePort
    using ModelingToolkit: t_nounits as t, D_nounits as D, @named, @variables, @parameters, @component, System, Equation, SymbolicT, extend, @unpack
    using Symbolics: unwrap

    # ==========================================
    # 1. Morris-Lecar (Built via GenericChannel)
    # ==========================================
    
    # Fast Ca2+ gating (effectively instantaneous)
    const V1, V2 = -20.0, 15.0
    const ml_ca_m = v -> (
        0.5 .* (1.0 .+ tanh.((v .- V1) ./ V2)) ./ 0.1,
        0.5 .* (1.0 .- tanh.((v .- V1) ./ V2)) ./ 0.1
    )

    # Slow K+ gating (recovery variable)
    const V3, V4 = -25.0, 5.0
    const tau_n = 10.0
    const ml_k_n = v -> (
        0.5 .* (1.0 .+ tanh.((v .- V3) ./ V4)) ./ tau_n,
        0.5 .* (1.0 .- tanh.((v .- V3) ./ V4)) ./ tau_n
    )

    @component function MorrisLecar(; name, topology=Scalar(), V_init=-20.0, 
                          g_Ca=4.0, E_Ca=100.0, g_K=8.5, E_K=-70.0, g_L=0.1, E_L=-50.0)
        m0 = 0.5 * (1 + tanh((V_init - V1) / V2))
        n0 = 0.5 * (1 + tanh((V_init - V3) / V4))
        
        ca_gates = [GateSpec(:m, 1, m0, ml_ca_m)]
        k_gates  = [GateSpec(:n, 1, n0, ml_k_n)]
        
        # Note: In a real build script, you'd create the Capacitor separately, 
        # but for convenience we can just document the required channels.
        # We return a tuple of the channels to be used with build_compartment.
        @named ca_ch = GenericChannel(topology=topology, g=g_Ca, E_rev=E_Ca, gates=ca_gates)
        @named k_ch  = GenericChannel(topology=topology, g=g_K, E_rev=E_K, gates=k_gates)
        @named leak  = GenericChannel(topology=topology, g=g_L, E_rev=E_L, gates=GateSpec[])
        
        return (ca_ch, k_ch, leak)
    end

    # ==========================================
    # 2. FitzHugh-Nagumo (Custom 2D OnePort)
    # ==========================================
    
    @component function FitzHughNagumo(; name, topology=Scalar(), I_ext=0.0, a=0.7, b=0.8, c=10.0, tau=12.5)
        if topology isa Scalar
            @named oneport = OnePort()
            @unpack v, i = oneport
            
            @parameters a=a b=b c=c tau=tau
            params = SymbolicT[a, b, c, tau]
            
            @variables w(t)=0.0
            vars = SymbolicT[v, w]
            
            # The channel provides the cubic and recovery dynamics.
            # C * dV/dt = I_ext - i_channel
            # We want: dV/dt = c * (v - v^3/3 - w) + I_ext
            # So: i_channel = -c * (v - v^3/3 - w)
            eqs = Equation[
                i ~ -c * (v - (v^3)/3.0 - w),
                D(w) ~ (v + a - b * w) / tau
            ]
            
            return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        else
            N = topology.N
            @named oneport = VectorizedOnePort(N=N)
            @unpack v, i = oneport
            
            @parameters a=a b=b c=c tau=tau
            params = SymbolicT[a, b, c, tau]
            
            @variables w(t)[1:N]=zeros(N)
            vars = SymbolicT[v, w]
            
            eqs = Equation[
                i ~ -c .* (v .- (v.^3)./ 3.0 .- w),
                D(w) ~ (v .+ a .- b .* w) ./ tau
            ]
            
            return extend(System(eqs, t, vars, params; systems=System[], name=name), oneport)
        end
    end

    export MorrisLecar, FitzHughNagumo
end

# ==========================================
# Standard Model Library
# ==========================================
module HodgkinHuxley
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, Scalar, Vectorized
    using ModelingToolkit: t_nounits as t, @named

    # Standard 1952 Hodgkin-Huxley gating dynamics (Dayan & Abbott formulation)
    # where V_rest = -65 mV.
    const na_m = v -> (
        0.1 .* (v .+ 40.0) ./ (1.0 .- exp.(-(v .+ 40.0) ./ 10.0)),  # alpha_m
        4.0 .* exp.(-(v .+ 65.0) ./ 18.0)                           # beta_m
    )
    const na_h = v -> (
        0.07 .* exp.(-(v .+ 65.0) ./ 20.0),                         # alpha_h
        1.0 ./ (1.0 .+ exp.(-(v .+ 35.0) ./ 10.0))                  # beta_h
    )
    const k_n = v -> (
        0.01 .* (v .+ 55.0) ./ (1.0 .- exp.(-(v .+ 55.0) ./ 10.0)), # alpha_n
        0.125 .* exp.(-(v .+ 65.0) ./ 80.0)                         # beta_n
    )

    # Steady-state initial conditions at V = -65 mV
    const sodium_gates = [GateSpec(:m, 3, 0.052, na_m), GateSpec(:h, 1, 0.596, na_h)]
    const potassium_gates = [GateSpec(:n, 4, 0.317, k_n)]

    # Convenience constructors
    function SodiumChannel(; name, topology=Scalar(), g=120.0, E_rev=50.0)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=sodium_gates, topology=topology)
    end

    function PotassiumChannel(; name, topology=Scalar(), g=36.0, E_rev=-77.0)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=potassium_gates, topology=topology)
    end

    function LeakChannel(; name, topology=Scalar(), g=0.3, E_rev=-54.4)
        return GenericChannel(; name=name, g=g, E_rev=E_rev, gates=GateSpec[], topology=topology)
    end

    export SodiumChannel, PotassiumChannel, LeakChannel
end

# ==========================================
# Standard Model Library
# ==========================================
module HodgkinHuxley
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, Scalar, Vectorized
    using ModelingToolkit: t_nounits as t, @named

    # Are these standard 1952 HH Gate Definitions? Forget where i found them. Check
    const na_m = v -> (
        0.182 .* (v .+ 35.0) ./ (1.0 .- exp.(-(v .+ 35.0) ./ 9.0)),
        -0.124 .* (v .+ 35.0) ./ (1.0 .- exp.((v .+ 35.0) ./ 9.0))
    )
    const na_h = v -> (
        0.25 .* exp.(-(v .+ 90.0) ./ 12.0),
        0.25 .* (exp.((v .+ 62.0) ./ 6.0)) ./ exp.(-(v .+ 90.0) ./ 12.0)
    )
    const k_n = v -> (
        0.02 .* (v .- 25.0) ./ (1.0 .- exp.(-(v .- 25.0) ./ 9.0)),
        -0.002 .* (v .- 25.0) ./ (1.0 .- exp.((v .- 25.0) ./ 9.0))
    )

    const sodium_gates = [GateSpec(:m, 3, 0.0, na_m), GateSpec(:h, 1, 0.0, na_h)]
    const potassium_gates = [GateSpec(:n, 4, 0.0, k_n)]

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



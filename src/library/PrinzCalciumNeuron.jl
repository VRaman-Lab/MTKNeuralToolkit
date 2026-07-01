module PrinzNeuron
    using ..MTKNeuralToolkit: GateSpec, GenericChannel, CaVChannel, KCaChannel, CalciumTracker, Capacitor, build_compartment, Scalar
    import ..MTKNeuralToolkit: AbstractGeometry, get_capacitance, get_conductance, get_ca_conversion_factor, get_synaptic_conductance
    import ..MTKNeuralToolkit: InfTau, InfTauCa
    using ModelingToolkit: @named


    # Prinz uses a custom conversion factor to go from geometry to calcium flow so we recreate it
    Base.@kwdef struct PrinzGeometry <: AbstractGeometry
        C_m::Float64 = 10.0
        area::Float64 = 0.0628
    end

    # Replicate the exact math of the original script
    get_capacitance(C, geom::PrinzGeometry) = geom.C_m
    get_conductance(g, geom::PrinzGeometry) = g * (geom.C_m / geom.area)
    get_ca_conversion_factor(conv, geom::PrinzGeometry, tauCa) = 0.94 / (geom.C_m * tauCa)
    get_synaptic_conductance(g, geom::PrinzGeometry) = g * (1e-3 / geom.area^2)

 
    

    # 1. Define Inf and Tau functions based on Prinz equations
    # Na channel
    Na_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 25.5) ./ -5.29))
    Na_tau_m(v) = 2.64 .- 2.52 ./ (1 .+ exp.((v .+ 120.0) ./ -25.0))
    Na_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 48.9) ./ 5.18))
    Na_tau_h(v) = (1.34 ./ (1.0 .+ exp.((v .+ 62.9) ./ -10.0))) .* (1.5 .+ 1.0 ./ (1.0 .+ exp.((v .+ 34.9) ./ 3.6)))
    const na_gates = [GateSpec(:mNa, 3, 0.0, InfTau(Na_m_inf, Na_tau_m)), 
                      GateSpec(:hNa, 1, 0.0, InfTau(Na_h_inf, Na_tau_h))]

    # CaS channel
    CaS_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 33.0) ./ -8.1))
    CaS_tau_m(v) = 2.8 .+ 14.0 ./ (exp.((v .+ 27.0) ./ 10.0) .+ exp.((v .+ 70.0) ./ -13.0))
    CaS_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 60.0) ./ 6.2))
    CaS_tau_h(v) = 120.0 .+ 300.0 ./ (exp.((v .+ 55.0) ./ 9.0) .+ exp.((v .+ 65.0) ./ -16.0))
    const cas_gates = [GateSpec(:mCaS, 3, 0.0, InfTau(CaS_m_inf, CaS_tau_m)), 
                       GateSpec(:hCaS, 1, 0.0, InfTau(CaS_h_inf, CaS_tau_h))]

    # CaT channel
    CaT_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 27.1) ./ -7.2))
    CaT_tau_m(v) = 43.4 .- 42.6 ./ (1.0 .+ exp.((v .+ 68.1) ./ -20.5))
    CaT_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 32.1) ./ 5.5))
    CaT_tau_h(v) = 210.0 .- 179.6 ./ (1.0 .+ exp.((v .+ 55.0) ./ -16.9))
    const cat_gates = [GateSpec(:mCaT, 3, 0.0, InfTau(CaT_m_inf, CaT_tau_m)), 
                       GateSpec(:hCaT, 1, 0.0, InfTau(CaT_h_inf, CaT_tau_h))]

    # Ka channel
    Ka_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 27.2) ./ -8.7))
    Ka_tau_m(v) = 23.2 .- 20.8 ./ (1.0 .+ exp.((v .+ 32.9) ./ -15.2))
    Ka_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 56.9) ./ 4.9))
    Ka_tau_h(v) = 77.2 .- 58.4 ./ (1.0 .+ exp.((v .+ 38.9) ./ -26.5))
    const ka_gates = [GateSpec(:mKa, 3, 0.0, InfTau(Ka_m_inf, Ka_tau_m)), 
                     GateSpec(:hKa, 1, 0.0, InfTau(Ka_h_inf, Ka_tau_h))]

    # KCa channel
    KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.((v .+ 28.3) ./ -12.6))
    KCa_tau_m(v) = 180.6 .- 150.2 ./ (1.0 .+ exp.((v .+ 46.0) ./ -22.7))
    const kca_gates = [GateSpec(:mKCa, 4, 0.0, InfTauCa(KCa_m_inf, KCa_tau_m))]

    # Kdr channel
    Kdr_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 12.3) ./ -11.8))
    Kdr_tau_m(v) = 14.4 .- 12.8 ./ (1.0 .+ exp.((v .+ 28.3) ./ -19.2))
    const kdr_gates = [GateSpec(:mKdr, 4, 0.0, InfTau(Kdr_m_inf, Kdr_tau_m))]

    # H channel
    H_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 75.0) ./ 5.5))
    H_tau_m(v) = 2.0 ./ (exp.((v .+ 169.7) ./ -11.6) .+ exp.((v .- 26.7) ./ 14.3))
    const h_gates = [GateSpec(:mH, 1, 0.0, InfTau(H_m_inf, H_tau_m))]

    # 2. Build function
    function build_prinz_neuron(; name=:Prinz_Neuron, tauCa=200.0, Ca_inf=0.05, V_init=-50.0,
                                 gNa=100.0, gCaS=4.0, gCaT=2.0, gKa=10.0, gKCa=5.0, gKdr=10.0, gH=0.1, gleak=0.01,
                                 ENa=50.0, EK=-80.0, EH=-20.0, Eleak=-50.0, geom=PrinzGeometry())
        top = Scalar()
        nernst_factor = 500.0 * 8.6174e-5 * 283.15

        # Pass geom and tauCa to everything! The dispatch handles the math.
        @named na_ch  = GenericChannel(topology=top, g=gNa, E_rev=ENa, gates=na_gates, geometry=geom)
        @named cas_ch = CaVChannel(topology=top, g=gCaS, gates=cas_gates, Ca_out=3000.0, 
                                   nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
        @named cat_ch = CaVChannel(topology=top, g=gCaT, gates=cat_gates, Ca_out=3000.0, 
                                   nernst_factor=nernst_factor, geometry=geom, tauCa=tauCa)
        @named ka_ch  = GenericChannel(topology=top, g=gKa, E_rev=EK, gates=ka_gates, geometry=geom)
        @named kca_ch = KCaChannel(topology=top, g=gKCa, E_rev=EK, gates=kca_gates, geometry=geom)
        @named kdr_ch = GenericChannel(topology=top, g=gKdr, E_rev=EK, gates=kdr_gates, geometry=geom)
        @named h_ch   = GenericChannel(topology=top, g=gH, E_rev=EH, gates=h_gates, geometry=geom)
        @named leak   = GenericChannel(topology=top, g=gleak, E_rev=Eleak, gates=GateSpec[], geometry=geom)

        @named cap = Capacitor(topology=top, geometry=geom)
        channels = [na_ch, cas_ch, cat_ch, ka_ch, kca_ch, kdr_ch, h_ch, leak]

        decay_fn = ca -> (Ca_inf .- ca) ./ tauCa
        ion_config = CalciumTracker(decay=decay_fn, Ca_init=Ca_inf)
        comp = build_compartment(cap, channels; name=name, V_init=V_init, topology=top, ion_config=ion_config)
        
        return comp
    end

    export build_prinz_neuron, PrinzGeometry, na_gates, cas_gates, cat_gates, ka_gates, kca_gates, kdr_gates, h_gates
end



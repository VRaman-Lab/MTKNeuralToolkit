using MTKNeuralToolkit
using ModelingToolkit: mtkcompile, @named
using OrdinaryDiffEq, Plots

# Convenience functions to map (inf, tau) -> (alpha, beta)
InfTau(inf_fn, tau_fn) = v -> (inf_fn(v) ./ tau_fn(v), (1.0 .- inf_fn(v)) ./ tau_fn(v))
InfTauCa(inf_fn, tau_fn) = (v, ca) -> (inf_fn(v, ca) ./ tau_fn(v), (1.0 .- inf_fn(v, ca)) ./ tau_fn(v))

top = Scalar()

function build_calcium_neuron()
    # 1. Define Inf and Tau functions (fully vectorized for future-proofing)
    Na_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 25.5) ./ -5.29))
    Na_tau_m(v) = 1.32 .- 1.26 ./ (1 .+ exp.((v .+ 120.0) ./ -25.0))
    
    Na_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 48.9) ./ 5.18))
    Na_tau_h(v) = (0.67 ./ (1.0 .+ exp.((v .+ 62.9) ./ -10.0))) .* (1.5 .+ 1.0 ./ (1.0 .+ exp.((v .+ 34.9) ./ 3.6)))
    
    CaS_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 33.0) ./ -8.1))
    CaS_tau_m(v) = 1.4 .+ 7.0 ./ (exp.((v .+ 27.0) ./ 10.0) .+ exp.((v .+ 70.0) ./ -13.0))
    
    CaS_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 60.0) ./ 6.2))
    CaS_tau_h(v) = 60.0 .+ 150.0 ./ (exp.((v .+ 55.0) ./ 9.0) .+ exp.((v .+ 65.0) ./ -16.0))
    
    CaT_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 27.1) ./ -7.2))
    CaT_tau_m(v) = 21.7 .- 21.3 ./ (1.0 .+ exp.((v .+ 68.1) ./ -20.5))
    
    CaT_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 32.1) ./ 5.5))
    CaT_tau_h(v) = 105.0 .- 89.8 ./ (1.0 .+ exp.((v .+ 55.0) ./ -16.9))
    
    Ih_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 70.0) ./ 6.0))
    Ih_tau_m(v) = (272.0 .+ 1499.0 ./ (1.0 .+ exp.((v .+ 42.2) ./ -8.73)))
    
    Ka_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 27.2) ./ -8.7))
    Ka_tau_m(v) = 11.6 .- 10.4 ./ (1.0 .+ exp.((v .+ 32.9) ./ -15.2))
    
    Ka_h_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 56.9) ./ 4.9))
    Ka_tau_h(v) = 38.6 .- 29.2 ./ (1.0 .+ exp.((v .+ 38.9) ./ -26.5))
    
    KCa_m_inf(v, ca) = (ca ./ (ca .+ 3.0)) ./ (1.0 .+ exp.((v .+ 28.3) ./ -12.6))
    KCa_tau_m(v) = 90.3 .- 75.1 ./ (1.0 .+ exp.((v .+ 46.0) ./ -22.7))
    
    Kdr_m_inf(v) = 1.0 ./ (1.0 .+ exp.((v .+ 12.3) ./ -11.8))
    Kdr_tau_m(v) = 7.2 .- 6.4 ./ (1.0 .+ exp.((v .+ 28.3) ./ -19.2))

    # 2. Define Gates using the convenience functions
    na_gates  = [GateSpec(:mNa, 3, 0.0, InfTau(Na_m_inf, Na_tau_m)), 
                 GateSpec(:hNa, 1, 0.0, InfTau(Na_h_inf, Na_tau_h))]
    
    cas_gates = [GateSpec(:mCaS, 3, 0.0, InfTau(CaS_m_inf, CaS_tau_m)), 
                 GateSpec(:hCaS, 1, 0.0, InfTau(CaS_h_inf, CaS_tau_h))]
    
    cat_gates = [GateSpec(:mCaT, 3, 0.0, InfTau(CaT_m_inf, CaT_tau_m)), 
                 GateSpec(:hCaT, 1, 0.0, InfTau(CaT_h_inf, CaT_tau_h))]
    
    ih_gates  = [GateSpec(:mIh, 1, 0.0, InfTau(Ih_m_inf, Ih_tau_m))]
    
    ka_gates  = [GateSpec(:mKa, 3, 0.0, InfTau(Ka_m_inf, Ka_tau_m)), 
                 GateSpec(:hKa, 1, 0.0, InfTau(Ka_h_inf, Ka_tau_h))]
    
    kca_gates = [GateSpec(:mKCa, 4, 0.0, InfTauCa(KCa_m_inf, KCa_tau_m))]
    
    kdr_gates = [GateSpec(:mKdr, 4, 0.0, InfTau(Kdr_m_inf, Kdr_tau_m))]

    # 3. Create Channels
    nernst_factor = 500.0 * 8.6174e-5 * 283.15
    
    @named na_ch  = GenericChannel(topology=top, g=100.0, E_rev=50.0, gates=na_gates)
    @named cas_ch = CaVChannel(topology=top, g=3.0, conversion_factor=0.94, gates=cas_gates, Ca_out=3000.0, nernst_factor=nernst_factor)
    @named cat_ch = CaVChannel(topology=top, g=1.3, conversion_factor=0.94, gates=cat_gates, Ca_out=3000.0, nernst_factor=nernst_factor)
    @named ih_ch  = GenericChannel(topology=top, g=0.5, E_rev=-20.0, gates=ih_gates)
    @named ka_ch  = GenericChannel(topology=top, g=5.0, E_rev=-80.0, gates=ka_gates)
    @named kca_ch = KCaChannel(topology=top, g=10.0, E_rev=-80.0, gates=kca_gates)
    @named kdr_ch = GenericChannel(topology=top, g=20.0, E_rev=-80.0, gates=kdr_gates)
    @named leak   = GenericChannel(topology=top, g=0.01, E_rev=-50.0, gates=GateSpec[])
    
    @named cap = Capacitor(topology=top, C=1.0)
    channels = [na_ch, cas_ch, cat_ch, ih_ch, ka_ch, kca_ch, kdr_ch, leak]

    # 4. Calcium Pool Configuration
    decay_fn = ca -> (0.05 .- ca) ./ 20.0
    ion_config = CalciumTracker(decay=decay_fn, Ca_init=0.05)

    comp = build_compartment(cap, channels; name=:Liu_AB_Neuron, V_init=-60.0, topology=top, ion_config=ion_config)
    
    # Provide a small driver just to kick off the burst
    drivers = [(1, 10.0)]
    net = build_acausal_network([comp]; drivers=drivers, name=:AB_network)
    
    return net
end

# --- Simulation ---
net = build_calcium_neuron()
net_compiled = mtkcompile(net.sys)

# Rosenbrock23() handles the stiff calcium dynamics nicely
prob = ODEProblem(net_compiled, [], (0.0, 500.0))
sol = solve(prob, Rosenbrock23())

# === Plot ===
p1 = plot(sol, idxs=net_compiled.Liu_AB_Neuron.cap.v, title="Membrane Potential", legend=false)
p2 = plot(sol, idxs=net_compiled.Liu_AB_Neuron.Liu_AB_Neuron_ca_pool.Ca, title="Calcium Concentration", legend=false)

plot(p1, p2, layout=(2,1), size=(800,500))

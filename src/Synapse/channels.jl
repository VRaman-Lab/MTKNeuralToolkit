@mtkmodel BaseSynapse begin
    @extend v_pre, v_post, i_post = twoport = DirectionalTwoPort()
    @parameters begin
        g, [description = "Conductance"]
        E, [description = "Reversal potential"]
        Vth, [description = "Threshold voltage"]
        k_, [description = "Kinetic parameter"]
        sigma, [description = "Sigmoid steepness"]
    end
    @variables begin
        s_hat(t)
        tau_s(t)
        s(t) = 0.0
    end
    @equations begin
        s_hat ~ 1.0 / (1.0 + exp((Vth - v_pre) / sigma))
        tau_s ~ (1.0 - s_hat) / k_
        D(s) ~ (1/tau_s) * (s_hat - s)
        i_post ~ g * s * (v_post - E)
    end
end

@mtkmodel CholinergicSynapse begin
    @extend v_pre, v_post, i_post = twoport = DirectionalTwoPort()
    @parameters begin
        g, [description = "Conductance"]
        E = -80.0, [description = "Reversal potential"]
        Vth = -35.0, [description = "Threshold voltage"]
        k_ = 0.01, [description = "Kinetic parameter"]
        sigma = 5.0, [description = "Sigmoid steepness"]
    end
    @variables begin
        s_hat(t)
        tau_s(t)
        s(t) = 0.0
    end
    @equations begin
        s_hat ~ 1.0 / (1.0 + exp((Vth - v_pre) / sigma))
        tau_s ~ (1.0 - s_hat) / k_
        D(s) ~ (1/tau_s) * (s_hat - s)
        i_post ~ g * s * (v_post - E)
    end
end

@mtkmodel GlutamatergicSynapse begin
    @extend v_pre, v_post, i_post = twoport = DirectionalTwoPort()
    @parameters begin
        g, [description = "Conductance"]
        E = -70.0, [description = "Reversal potential"]
        k_ = 0.025, [description = "Kinetic parameter"]
        Vth = -35.0, [description = "Threshold voltage"]
        sigma = 5.0, [description = "Sigmoid steepness"]
    end
    @variables begin
        s(t) = 0.0
        s_hat(t)
        tau_s(t)
    end
    @equations begin
        s_hat ~ 1.0 / (1.0 + exp((Vth - v_pre) / sigma))
        tau_s ~ (1.0 - s_hat) / k_
        D(s) ~ (1/tau_s) * (s_hat - s)
        i_post ~ g * s * (v_post - E)
    end
end

@mtkmodel LifSynapse begin
    @extend v_pre, v_post, i_post = twoport = DirectionalTwoPort()
    @parameters begin
        g_max = 1.0 
        τ_s = 5.0 
        V_th = 10.0 
        A = 1.0
    end
    @variables begin
        s(t) 
    end
    @equations begin
        D(s) ~ -s / τ_s
        i_post ~ g_max * s * (v_pre - v_post)
    end
    @continuous_events begin
        [v_pre ~ V_th] => [s ~ s + A]
    end
end

E_syn_gate_preset(; g, E=0.0, Vth=-35.0, k_=0.025, sigma=5.0, name=:E_syn) = 
    BaseSynapse(; g=g, E=E, Vth=Vth, k_=k_, sigma=sigma, name=name)

I_syn_gate_preset(; g, E=-70.0, Vth=-35.0, k_=0.01, sigma=5.0, name=:I_syn) = 
    BaseSynapse(; g=g, E=E, Vth=Vth, k_=k_, sigma=sigma, name=name)

function custom_synapse(; g, E, Vth, k_, sigma, name=:custom_syn)
    return BaseSynapse(; g=g, E=E, Vth=Vth, k_=k_, sigma=sigma, name=name)
end
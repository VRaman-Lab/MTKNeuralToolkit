@mtkmodel BaseSynapse begin
    @extend v_pre, v_post, i_post = oneport = DirectionalTwoPort()
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

CholinergicSynapse(; g, E=-80.0, Vth=-35.0, k_=0.01, sigma=5.0, name=:cholinergic_syn) = 
    BaseSynapse(; g=g, E=E, Vth=Vth, k_=k_, sigma=sigma, name=name)

GlutamatergicSynapse(; g, E=-70.0, Vth=-35.0, k_=0.025, sigma=5.0, name=:glutamatergic_syn) = 
    BaseSynapse(; g=g, E=E, Vth=Vth, k_=k_, sigma=sigma, name=name)

"
LIF synapse: Allows current to be injected from pro-synapse to post-synapse
"
@mtkmodel LifSynapseComplex begin
    @extend v_pre, v_post, i_post = twoport = DirectionalTwoPort()
    @parameters begin
        g_max = 1.0 
        E = 0
        τ_g = 5.0
        V_th = -55.0
        k = 1
    end
    @variables begin 
        g(t) = 0.0 
    end 
    @equations begin
        τ_g * D(g) ~ -g + g_max / (1 + exp(-(v_pre - V_th)/k)) 
        i_post ~ (g * (v_post))
    end
end


E_syn_gate_preset(; g, E=0.0, Vth=-35.0, k_=0.025, sigma=5.0, name=:E_syn) = 
    BaseSynapse(; g=g, E=E, Vth=Vth, k_=k_, sigma=sigma, name=name)

I_syn_gate_preset(; g, E=-70.0, Vth=-35.0, k_=0.01, sigma=5.0, name=:I_syn) = 
    BaseSynapse(; g=g, E=E, Vth=Vth, k_=k_, sigma=sigma, name=name)

function custom_synapse(; g, E, Vth, k_, sigma, name=:custom_syn)
    return BaseSynapse(; g=g, E=E, Vth=Vth, k_=k_, sigma=sigma, name=name)
end
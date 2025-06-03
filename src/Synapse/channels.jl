@mtkmodel E_syn_gates begin
    @extend v_pre, v_post, i_post = twoport = DirectionalTwoPort()
    @parameters begin
        g, [description = "Conductance"]
        E = 0.0
        Vth = -35.0
        k_ = 0.025
        sigma = 5.0
    end
    @variables begin
        s_hat(t)
        tau_s(t)
        s(t)
    end
    @equations begin
        s_hat ~ 1.0 / (1.0 + exp((Vth - v_pre) / sigma))
        tau_s ~ (1.0 - s_hat) / k_
        D(s) ~ (1/tau_s) * (s_hat - s)
        i_post ~ g * s * (v_post - E)
    end
end

@mtkmodel I_syn_gates begin
    @extend v_pre, v_post, i_post = twoport = DirectionalTwoPort()
    @parameters begin
        g, [description = "Conductance"]
        E = -70.0
        Vth = -35.0
        k_ = 0.01
        sigma = 5.0
    end
    @variables begin
        s_hat(t)
        tau_s(t)
        s(t)
    end
    @equations begin
        s_hat ~ 1.0 / (1.0 + exp((Vth - v_pre) / sigma))
        tau_s ~ (1.0 - s_hat) / k_
        D(s) ~ (1/tau_s) * (s_hat - s)
        i_post ~ g * s * (v_post - E)
    end
end

@mtkmodel IF_channel begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        V_rest = -65.0
        V_reset = -70.0
        V_th = -55.0
        τ_m = 10.0     # Membrane time constant
        R = 1.0
        C = 10.0
    end
    @equations begin
        i ~ (v - V_rest)/R + C*D(v)
    end 
    @continuous_events begin
        [v ~ V_th] => [v ~ Pre(v) - 10]
    end
end



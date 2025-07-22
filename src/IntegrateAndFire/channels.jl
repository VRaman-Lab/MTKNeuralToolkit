@mtkmodel IF_channel begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        E
        V_reset = -2.0
        V_th = 10.0
        τ_m = 1.0     # Membrane time constant
        R = 1.0
        C = 10.0
    end
    @equations begin
        i ~ (v - E - 65)/R + C*D(v)
    end 
    @continuous_events begin
        [v ~ V_th] => [v ~ V_reset]
    end
end



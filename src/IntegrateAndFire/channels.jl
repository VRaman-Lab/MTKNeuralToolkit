@mtkmodel LIF begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        E
        V_reset = -2.0
        V_th = 10.0
        τ_m = 1.0 
        R = 1.0
        C = 10.0
    end
    @equations begin
        i ~ (v - E)/R + C*D(v)
    end 
    @continuous_events begin
        [v ~ V_th] => [v ~ V_reset]
    end
end

@mtkmodel IF begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        E
        V_reset = -2.0
        V_th = 10.0
        τ_m = 1.0 
        R = 1.0
        C = 10.0
    end
    @equations begin
        i ~ (v - E)/R 
    end 
    @continuous_events begin
        [v ~ V_th] => [v ~ V_reset]
    end
end



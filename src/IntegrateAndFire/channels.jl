@mtkmodel IF_channel begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        R = 2
    end
    @equations begin
        i ~ v/R
    end 
end

IF_Channel(; name=:conductance, kwargs...) =  IF_channel( ;name, kwargs...)





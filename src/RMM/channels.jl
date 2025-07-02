@mtkmodel rmmvec begin
    @extend v, i = oneport = OnePort()
    #@variables lti_v[1:8](t) #Doesn't work :(
    @variables begin
        lti_v₁(t) = default_ltiv
        lti_v₂(t) = default_ltiv
        lti_v₃(t) = default_ltiv
        lti_v₄(t) = default_ltiv
        lti_v₅(t) = default_ltiv
        lti_v₆(t) = default_ltiv
        lti_v₇(t) = default_ltiv
        lti_v₈(t) = default_ltiv
        lti_v_plotter(t)
    end
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
        A_Mat[1:8, 1:8]::Float64
        B_Vec[1:8]::Float64
        default_ltiv = -70
        default_activation_function = tanh
        default_n_inputs = 8
        default_n_outputs = 8
        default_width = 16
        default_depth = 1
    end
    @components begin
        nn_in = RealInputArray(nin = default_n_inputs)
        nn_out = RealOutputArray(nout = default_n_outputs)
        nn = NeuralNetworkBlock(n_input = default_n_inputs, n_output = default_n_outputs; 
                                chain = multi_layer_feed_forward(default_n_inputs, default_n_outputs, width=default_width, depth = default_depth, activation=default_activation_function), rng=Xoshiro(57))
    end
    @equations begin        
        D(lti_v₁) ~ A_Mat[1,1] * lti_v₁ + B_Vec[1] * v
        D(lti_v₂) ~ A_Mat[2,2] * lti_v₂ + B_Vec[2] * v
        D(lti_v₃) ~ A_Mat[3,3] * lti_v₁ + B_Vec[3] * v
        D(lti_v₄) ~ A_Mat[4,4] * lti_v₁ + B_Vec[4] * v
        D(lti_v₅) ~ A_Mat[5,5] * lti_v₁ + B_Vec[5] * v
        D(lti_v₆) ~ A_Mat[6,6] * lti_v₁ + B_Vec[6] * v
        D(lti_v₇) ~ A_Mat[7,7] * lti_v₁ + B_Vec[7] * v
        D(lti_v₈) ~ A_Mat[8,8] * lti_v₁ + B_Vec[8] * v
        nn_in.u ~ [lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈]
        lti_v_plotter ~ sum([lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈])
        connect(nn_in, nn.input)
        connect(nn_out, nn.output)
        i ~ g * sum(nn.output) * (v-E)
    end   
end
function rmmvecf(; name=:conductance, n_inputs=8, n_outputs=8, width=16, depth=1, 
                activation=tanh, default_ltiv=-70, kwargs...)
    
    @mtkmodel rmmvec_instance begin
        @extend v, i = oneport = OnePort()
        @variables begin
            lti_v₁(t) = default_ltiv
            lti_v₂(t) = default_ltiv
            lti_v₃(t) = default_ltiv
            lti_v₄(t) = default_ltiv
            lti_v₅(t) = default_ltiv
            lti_v₆(t) = default_ltiv
            lti_v₇(t) = default_ltiv
            lti_v₈(t) = default_ltiv
            lti_v_plotter(t)
        end
        @parameters begin
            g = 0.01, [description = "Conductance"]
            E = -65.0
            A_Mat[1:8]::Float64
            B_Vec[1:8]::Float64
        end
        @components begin
            nn_in = RealInputArray(nin = n_inputs)  
            nn_out = RealOutputArray(nout = n_outputs)
            nn = NeuralNetworkBlock(n_input = n_inputs, n_output = n_outputs; 
                                    chain = multi_layer_feed_forward(n_inputs, n_outputs, 
                                                                    width=width, depth=depth, 
                                                                    activation=activation), 
                                    rng=Xoshiro(57))
        end
        @equations begin        
            D(lti_v₁) ~ A_Mat[1] * lti_v₁ + B_Vec[1] * v
            D(lti_v₂) ~ A_Mat[2] * lti_v₂ + B_Vec[2] * v
            D(lti_v₃) ~ A_Mat[3] * lti_v₃ + B_Vec[3] * v  
            D(lti_v₄) ~ A_Mat[4] * lti_v₄ + B_Vec[4] * v
            D(lti_v₅) ~ A_Mat[5] * lti_v₅ + B_Vec[5] * v
            D(lti_v₆) ~ A_Mat[6] * lti_v₆ + B_Vec[6] * v
            D(lti_v₇) ~ A_Mat[7] * lti_v₇ + B_Vec[7] * v
            D(lti_v₈) ~ A_Mat[8] * lti_v₈ + B_Vec[8] * v
            nn_in.u ~ [lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈]
            lti_v_plotter ~ sum([lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈])
            connect(nn_in, nn.input)
            connect(nn_out, nn.output)
            i ~ g * sum(nn.output.u) * (v-E)
        end   
    end
    
    return rmmvec_instance(; name, kwargs...)
end

@mtkmodel rmmscal begin
    @extend v, i = oneport = OnePort()
    @variables begin
        lti_v(t) = 0.0
    end
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
        A_Mat = 0.6065
        B_Vec = 0.3935
    end
    @components begin
        nn_in = RealInputArray(nin = 1)
        nn_out = RealOutputArray(nout = 1)
        nn = NeuralNetworkBlock(n_input = 1, n_output = 1; 
                                chain = multi_layer_feed_forward(1, 1, width=4), rng=Xoshiro(57))
    end
    @equations begin        
        D(lti_v) ~ A_Mat * lti_v + B_Vec * v
        nn_in.u[1] ~ lti_v   
        connect(nn_in, nn.output)
        connect(nn_out, nn.input)
        i ~ g * nn.output.u[1]
    end
end


RMMVec(;name=:conductance, kwargs...) = rmmvec(;name, kwargs...)
RMMVecf(;name=:conductance, kwargs...) = rmmvecf(;name, kwargs...)
RMMScal(;name=:conductance, kwargs...) = rmmscal(;name, kwargs...)
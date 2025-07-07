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
            nn_in.u ~ lti_min_max_norm([lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈])
            lti_v_plotter ~ sum([lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈])
            connect(nn_in, nn.input)
            connect(nn_out, nn.output)
            i ~ g * sum(nn.output.u) * (v-E)
        end   
    end
    
    return rmmvec_instance(; name, kwargs...)
end

function rmmvecf_v3(;τ::Vector, name=:conductance, tf=0.01, n_outputs=8, width=16, depth=1, 
                   activation=tanh, default_ltiv=-70, kwargs...)

    A_Mat, B_Vec = make_lti_vecs(τ;δ=tf)
    n_inputs = length(B_Vec)

    @named oneport = OnePort()
    @parameters begin
        t
        g=0.01
        E=-65.0
        A_Mat[1:n_inputs] = A_Mat
        B_Vec[1:n_inputs] = B_Vec
    end
    @variables begin
        v(t)
        i(t)
        lti_v(t)[1:n_inputs] = default_ltiv
        lti_v_plotter(t)
    end
    
    D = Differential(t)
    
    # Create neural network components directly
    @named nn_in = RealInputArray(nin = n_inputs)  
    @named nn_out = RealOutputArray(nout = n_outputs)
    @named nn = NeuralNetworkBlock(n_input = n_inputs, n_output = n_outputs; 
                                   chain = multi_layer_feed_forward(n_inputs, n_outputs, 
                                                                   width=width, depth=depth, 
                                                                   activation=activation), 
                                   rng=Xoshiro(57))
    
    # LTI differential equations
    lti_eqs = [D(lti_v[i]) ~ A_Mat[i] * lti_v[i] + B_Vec[i] * v for i in 1:n_inputs]
    
    # System equations
    sys_eqs = [
        v ~ oneport.v
        i ~ oneport.i
        lti_v_plotter ~ sum(lti_v)
        nn_in.u ~ lti_min_max_norm(collect(lti_v))
        connect(nn_in, nn.input)
        connect(nn_out, nn.output)
        i ~ g * sum(nn.output.u) * (v - E)
    ]
    
    eqs = vcat(lti_eqs, sys_eqs)
    
    sys = ODESystem(eqs, t, name=name,[v, i, lti_v..., lti_v_plotter], 
                          [g, E, A_Mat..., B_Vec...]; 
                          systems=[nn_in, nn_out, nn, oneport])
    
    return sys
end

RMMVec(;name=:conductance, kwargs...) = rmmvec(;name, kwargs...)
RMMVecf(;name=:conductance, kwargs...) = rmmvecf(;name, kwargs...)
RMMScal(;name=:conductance, kwargs...) = rmmscal(;name, kwargs...)
RMMVecf_v2(;name=:conductance, kwargs...) = rmmvecf_v2(;name, kwargs...)
RMMVecf_v3(;name=:conductance, kwargs...) = rmmvecf_v3(;name, kwargs...)
function rmmvecf(;τ::Vector, name=:conductance, tf=0.01, n_outputs=8, width=16, depth=1, 
                   activation=tanh, default_ltiv=-70, kwargs...)

    A_Mat, B_Vec = make_lti_vecs(τ;δ=tf)
    n_inputs = length(B_Vec)
    @named p = Pin()
    @named n = Pin()
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
        connect(oneport.p, p)
        connect(oneport.n, n)
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

RMMVecf(;name=:conductance, kwargs...) = rmmvecf(;name, kwargs...)
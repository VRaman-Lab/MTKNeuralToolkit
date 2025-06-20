#=@mtkmodel ANN begin
    @extend v, i = oneport = OnePort()
    @parameters begin
        dimensions = [], [description="ANN dimensions: a vector, with each element being the size of layer 1, 2, etc"]
        g
        E
    end
    #model = model_base()
    @parameters begin
        #=ANN_chain = Lux.Chain(
            Lux.Dense(1 => 8, Lux.mish, use_bias = false),
            Lux.Dense(8 => 8, Lux.mish, use_bias = false),
            Lux.Dense(8 => 1, use_bias = false)
        )
        nn = NeuralNetworkBlock(1, 1; 
            chain = ANN_chain, 
            rng = StableRNG(1111), name =:test)=#
        #chain = multi_layer_feed_forward(1, 1)
        @named nn = NeuralNetworkBlock(1,1)
    end
    @variables begin
    end
    @equations begin
        #connect(model.nn_in, nn.output)
        #connect(model.nn_out, nn.input)
        nn_input ~ v
        i ~ g * nn.output.u * (log(v-E,10))
    end
end=#

function ANN(; name)
    @variables v(t) i(t)
    @parameters g=1.0 E=-65.0
    
    # Create the neural network component manually
    chain = multi_layer_feed_forward(1, 1)
    @named nn = NeuralNetworkBlock(; n_input=1, n_output=1, 
                                   chain=chain, rng=Xoshiro(42))
    
    # Create OnePort manually
    @named oneport = OnePort()
    
    eqs = [
        v ~ oneport.v,
        i ~ oneport.i,
        connect(nn.input, [v]),  # Connect voltage to NN input
        i ~ g * nn.output.u * log10(v - E)
    ]
    
    return ODESystem(eqs, t, [v, i], [g, E]; 
                    systems=[oneport, nn], name=name)
end


#const ANN_Chain = Lux.Chain(Lux.Dense(1 => 16, Lux.mish, use_bias = false),Lux.Dense(16 => 8, Lux.mish, use_bias = false),Lux.Dense(8 => 1, use_bias = false))
chain = multi_layer_feed_forward(1, 1)
#=function model_base(v)
    @variables begin
        y(t) = 0.0
        nn_input(t)
        nn_output(t)
    end
    @named nn_in(t) = RealInput()
    @named nn_out(t) = RealOutput()
    @equations begin
        nn_in.u ~ nn_input
        nn_output ~ nn_out.u
        y ~ nn_output
    end
    return ODESystem(eqs, t, [y, nn_input, nn_output], [], name = :model_base, systems = [nn_in, nn_out])
end=#
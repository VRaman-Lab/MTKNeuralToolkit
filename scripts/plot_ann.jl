import Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))

using ModelingToolkitNeuralNets
using Lux
using ModelingToolkit
using ModelingToolkitStandardLibrary
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Constant, RealInput, TimeVaryingFunction, Sum, RealInputArray, RealOutputArray
using ModelingToolkit: t_nounits as t, D_nounits as D
using Random

using ModelingToolkit
using OrdinaryDiffEq
using MTKNeuralToolkit 
import MTKNeuralToolkit.HodgkinHuxley as HH
import MTKNeuralToolkit.RMM as RMM
import MTKNeuralToolkit
using Plots
using LinearAlgebra
#=
@mtkmodel ANNFred begin
    @extend v, i = oneport = OnePort()
    #@variables begin
    #    nn_out_filt(t) = 0.0
    #end
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
        τ = 1e-3
    end
    @components begin
        nn_in = RealInputArray(nin = 1)
        nn_out = RealOutputArray(nout = 1)
        nn = NeuralNetworkBlock(n_input = 1, n_output = 1; 
                                chain = multi_layer_feed_forward(1, 1, width=5))
    end
    @equations begin
        v ~ nn_in.u[1]
        
        connect(nn_in, nn.output)
        connect(nn_out, nn.input)
        #D(nn_out_filt) ~ nn_out.u[1] / τ
        i ~ g * nn_out.u[1] * v
    end
end

@mtkmodel RMMBob begin
    @extend v, i = oneport = OnePort()
    @variables begin
        #lti_v[1:8](t) = zeros(8)
        lti_v(t) = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    end
    @parameters begin
        #A_diag[1:8] = [0.6065, 0.8465, 0.9048, 0.9310, 0.9512, 0.9834, 0.9900, 0.9929]
        #B[1:8] = [0.3935, 0.1535, 0.0952, 0.0690, 0.0488, 0.0166, 0.0100, 0.0071]
    end
    @equations begin
    #[D(lti_v[j]) ~ A_diag[j] * lti_v[j] + B[j] * v for j in 1:8]...    
    D(lti_v) ~ 0.5*lti_v + v
    i ~ lti_v    
    end
end

@mtkmodel RMMTed begin
    @extend v, i = oneport = OnePort()
    #@variables begin
    #    nn_out_filt(t) = 0.0
    #end
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
    end
    @components begin
        nn_in = RealInputArray(nin = 8)
        nn_out = RealOutputArray(nout = 8)
        nn = NeuralNetworkBlock(n_input = 8, n_output = 8; 
                                chain = multi_layer_feed_forward(8, 8, width=5))
    end
    @equations begin        
        connect(nn_in, nn.output)
        connect(nn_out, nn.input)
        #D(nn_out_filt) ~ nn_out.u[1] / τ
        i ~ g * sum(nn_out.u) * v
    end
end
=#
@mtkmodel RMMGertha begin
    @extend v, i = oneport = OnePort()
    #@variables lti_v[1:8](t)
    @variables begin
        lti_v₁(t) = -5
        lti_v₂(t) = -5
        dnn_out(t) = -0.1
        #=lti_v₃(t) = -5.0
        lti_v₄(t) = -5.0
        lti_v₅(t) = -5.0
        lti_v₆(t) = -5.0
        lti_v₇(t) = -5.0
        lti_v₈(t) = -5.0=#
    end
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
        τ = 1e-6
        A_Small[1:2, 1:2]::Float64
        B_Small[1:2]::Float64
    end
    @components begin
        nn_in = RealInputArray(nin = 2)
        nn_out = RealOutputArray(nout = 2)
        nn = NeuralNetworkBlock(n_input = 2, n_output = 2; 
                                chain = multi_layer_feed_forward(2, 2, width=5), rng=Xoshiro(57))
    end
    @equations begin        
        #[D(lti_v[j]) ~ A_DIAG[j,j] * lti_v[j] + B_VEC[j] * v for j in 1:8]...
        #=D([lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈]) ~ 
        A_DIAG * [lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈] + B_VEC * v=#
        #nn_in.u ~ [lti_v₁, lti_v₂, lti_v₃, lti_v₄, lti_v₅, lti_v₆, lti_v₇, lti_v₈]
        #nn_in.u ~ lti_v
        D(lti_v₁) ~ A_Small[1,1] * lti_v₁ + B_Small[1] * v
        D(lti_v₂) ~ A_Small[2,2] * lti_v₂ + B_Small[2] * v
        nn_in.u ~ [lti_v₁, lti_v₂]
        #connect(nn_in, nn.output)
        #connect(nn_out, nn.input)
        connect(nn_in, nn.input)
        connect(nn_out, nn.output)
        D(dnn_out) ~ (sum(nn.output.u) - dnn_out) / τ
        i ~ g * dnn_out * (v-E)
    end    

end
A_Small = diagm([-0.6065, -0.8465])
B_Small = [0.3935, 0.1535]
#const A_DIAG1 = diagm([0.6065, 0.8465, 0.9048, 0.9310, 0.9512, 0.9834, 0.9900, 0.9929])
#const B_VEC1 = [0.3935, 0.1535, 0.0952, 0.0690, 0.0488, 0.0166, 0.0100, 0.0071]
#=
@mtkmodel RMMSlimjim begin
    @extend v, i = oneport = OnePort()
    #@variables begin
    #    nn_out_filt(t) = 0.0
    #end
    @variables begin
        lti_v(t) = 0.0
    end
    @parameters begin
        g = 0.01, [description = "Conductance"]
        E = -65.0
        A_diag = 0.6065
        B = 0.3935
    end
    @components begin
        nn_in = RealInputArray(nin = 1)
        nn_out = RealOutputArray(nout = 1)
        nn = NeuralNetworkBlock(n_input = 1, n_output = 1; 
                                chain = multi_layer_feed_forward(1, 1, width=5), rng=Xoshiro(57))
    end
    @equations begin        
        D(lti_v) ~ 0.1 * tanh((A_diag * lti_v + B * v) / 10.0)
        nn_in.u[1] ~ lti_v   
        connect(nn_in, nn.output)
        connect(nn_out, nn.input)
        #D(nn_out_filt) ~ nn_out.u[1] / τ
        i ~ g * nn_out.u[1]
    end
end=#
#LTI(;name=:LTI, kwargs...) = RMMBob(;name, kwargs...)
#ANN(;name=:ANN, kwargs...) = RMMTed(;name, kwargs...)
#lti = LTI()
#ann = ANN()
#RMM = build_RMM(lti, ann; name=:RMM)
#ANNGates(;name=:conductance, kwargs...) = ANNFred(;name, kwargs...)
RMMGates(;name=:condutance, kwargs...) = RMMGertha(;name, kwargs...)
Na =    build_channel(HH.NaGates(;g=40, E = 55), FixedReversal(;E=55); name = :Na)      
K =     build_channel(HH.KGates( ;g=35, E = -77), FixedReversal(;E=-77); name = :K)
Leak =  build_channel(HH.LGates( ;g=0.3, E = -65), FixedReversal(;E=-65); name = :Leak)
@named phanpy = RMMGates(g=0.1, E=-65,  A_Small=A_Small, B_Small=B_Small;name=:conductance)
donphan = build_channel(phanpy, FixedReversal(E=-77); name =:RMM)

@named inp = TimeVaryingFunction(f=t -> sin(t))
fn = BasicSoma(; C=1, name = :soma)
println("________________")
neur = build_neuron(fn, inp; channels = [donphan])
#=println("neur fields: ", fieldnames(typeof(neur)))
println("neur.eqs field type: ", typeof(neur.eqs))
println("neur.systems: ", neur.systems)
#print(ModelingToolkit.OptimizationSystem.equations(neur))=#
neur_c = structural_simplify(neur) 

#=u0 = [
    RMMGates.lti_v[1] = 0.0,
    RMMGates.lti_v[2] = 0.0,
    RMMGates.lti_v[3] = 0.0,
    RMMGates.lti_v[4] = 0.0,
    RMMGates.lti_v[5] = 0.0,
    RMMGates.lti_v[6] = 0.0,
    RMMGates.lti_v[7] = 0.0,
    RMMGates.lti_v[8] = 0.0,
]=#
prob = ODEProblem(neur_c, Pair[], (0.0, 200.0) )
println("Initial conditions: ", prob.u0)
println("Parameters: ", prob.p)
any(isnan.(prob.u0)) && println("NaN in initial conditions!")
any(isinf.(prob.u0)) && println("Inf in initial conditions!")
sol = solve(prob, Rodas5());
println("Solution status: ", sol.retcode)
println("Solution times: ", sol.t)
println("Number of timesteps: ", length(sol.t))
#plot(sol)
#p = plot(sol,idxs=[neur.Na.conductance.m_gate,neur.Na.conductance.h_gate], layout=(4,1), subplot=1)
#p = plot(sol,idxs=[neur.RMM.conductance.dnn_out], layout=(4,1), subplot=1)
#plot!(p, sol, idxs=[neur.K.conductance.n_gate], subplot=2)
#p = plot(sol,idxs=[neur.RMM.conductance.nn.input.u[1], neur.RMM.conductance.nn.output.u[1]], layout=(3,1), subplot=1)
#plot!(p, sol, idxs=[neur.RMM.conductance.nn.input.u[1], neur.RMM.conductance.nn.output.u[1]], subplot=2)
#plot!(p, sol, idxs=[neur.RMM.conductance.nn.input.u[2], neur.RMM.conductance.nn.output.u[2]], subplot=3)
plot!(p, sol, idxs=[neur.soma.v], subplot=2)

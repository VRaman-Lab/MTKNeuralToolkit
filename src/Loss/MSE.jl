"
This is all test code and is in the works (Ella dissertation)
"
function membrane_mse(system, sol, output_neurons)
    arrs = get_last_neurons_arrays(system, sol, output_neurons)
    weights = get_weights(sol, system)
    print(weights)
end



function get_last_neurons_arrays(system, solver, output_neurons)
    outputs  = []
    for name in output_neurons
        last_neuron = getproperty(getproperty(system, Symbol(name)), Symbol(name))
        print(last_neuron)
        Vm_Symbol = last_neuron.oneport.v
        push!(outputs, solver[Vm_Symbol, :])
    end
    return outputs
end 

function get_weights(sol, system)
    weight = getp(A, system.s12_LIF1.g_max)
    return weight
end 



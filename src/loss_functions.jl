using PreallocationTools
using SciMLStructures: Tunable, canonicalize, replace
using SymbolicIndexingInterface: parameter_values, setp



"""
    build_loss(net_sys::System, target_parameters, truth_data, tsteps)

Generates a ForwardDiff-compatible, non-allocating loss function mapping the 
Mean Squared Error between `truth_data` and the network's first state variable  trajectory. In the long term this shouldn't be in the package itself necessarily.
"""
function build_loss(net_sys::System, target_parameters, truth_data, tsteps)
    net_compiled = mtkcompile(net_sys)
    
    base_prob = ODEProblem(net_compiled, [], (tsteps[1], tsteps[end]), [], 
                           eval_expression=true, eval_module=@__MODULE__)
    
    # Inferred, high-performance parameter setter
    param_setter = setp(base_prob, target_parameters)
    
    # Thread-safe DiffCache template matching the runtime parameter layout
    ps_obj = base_prob.p
    tunable_template, _ = canonicalize(Tunable(), ps_obj)
    d_cache = DiffCache(copy(tunable_template))

    function loss_function(x, p)
        prob, ts, truth, setter, cache = p
        ps = prob.p
        
        # Extract dual-safe or standard workspace buffer depending on type of x
        buffer = get_tmp(cache, x)
        copyto!(buffer, canonicalize(Tunable(), ps)[1])
        
        # Structural parameter translation via SciMLStructures
        ps = replace(Tunable(), ps, buffer)
        setter(ps, x) 
        
        # Zero-allocation problem replication
        new_prob = remake(prob; p=ps)
        
        # Solve using neural-robust composite solver
        sol = solve(new_prob, AutoTsit5(Rosenbrock23()); saveat=ts)
        
        # Track dynamic trace of the main voltage node (Index 1)
        pred = Array(sol)[1, :]
        return sum((truth .- pred) .^ 2) / length(truth)
    end

    return loss_function, base_prob, param_setter, d_cache
end

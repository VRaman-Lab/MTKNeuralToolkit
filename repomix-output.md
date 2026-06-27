This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: src/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
src/
  BasicComponents.jl
  connections.jl
  loss_functions.jl
  MTKNeuralToolkit.jl
  tempgates.jl
```

# Files

## File: src/loss_functions.jl
````julia
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
````

## File: src/tempgates.jl
````julia
using Symbolics: variable
struct GateSpec
    name::Symbol
    power::Int
    ic::Float64
    # A function taking voltage `v` and returning a tuple: (alpha_expr, beta_expr)
    dynamics::Function 
end

@component function GenericChannel(; name, g, E_rev, gates::Vector{GateSpec})
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    
    # Dictionary to cleanly hold initial conditions for dynamically created vars
    init_conds = Dict{Any, Any}()
    
    conductance_factor = Num(1.0)
    
    for gate in gates
        # Dynamically create the gate variable and its rate variables
        gate_var = only(@variables $(gate.name)(t))
        alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
        beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
        
        push!(vars, gate_var, alpha_var, beta_var)
        init_conds[gate_var] = gate.ic
        
        # Call the user's function to get the symbolic alpha/beta equations
        alpha_expr, beta_expr = gate.dynamics(v)
        
        push!(eqs, alpha_var ~ alpha_expr)
        push!(eqs, beta_var ~ beta_expr)
        push!(eqs, D(gate_var) ~ alpha_var * (1 - gate_var) - beta_var * gate_var)
        
        # Multiply into the overall conductance (e.g., m^3 * h^1)
        conductance_factor *= gate_var ^ gate.power
    end
    
    # Final Ohm's law using driving force
    push!(eqs, i ~ g * conductance_factor * (v - E_rev))
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=System[], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end


@component function InlinedHHNeuron(; name, C=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0, E_L=-54.4, V_init=-65.0)
    @named oneport = OnePort()
    @unpack v, i, p, n = oneport
    @named injector = CurrentSource()
    @named ground = Ground()

    @parameters C=C g_Na=g_Na g_K=g_K g_L=g_L E_Na=E_Na E_K=E_K E_L=E_L
    params = SymbolicT[]
    push!(params, C, g_Na, g_K, g_L, E_Na, E_K, E_L)

    @variables begin
        V(t) = V_init
        m(t) = 0.0
        h(t) = 1.0
        n_gate(t) = 0.0
        I_Na(t)
        I_K(t)
        I_L(t)
        αₘ(t), βₘ(t)
        αₕ(t), βₕ(t)
        αₙ(t), βₙ(t)
    end
    vars = SymbolicT[]
    push!(vars, V, m, h, n_gate, I_Na, I_K, I_L, αₘ, βₘ, αₕ, βₕ, αₙ, βₙ)
    eqs = Equation[]
    push!(eqs, V ~ v)

    # Ground the membrane and the injector pins to prevent floating singularities
    push!(eqs, connect(ground.g, n))
    push!(eqs, connect(ground.g, injector.n))
    push!(eqs, connect(ground.g, injector.p))
    push!(eqs, i ~ p.i)

    # Na gating
    push!(eqs, αₘ ~ 0.182 * ((v - E_Na) + 35.0) / (1.0 - exp(-((v - E_Na) + 35.0) / 9.0)))
    push!(eqs, βₘ ~ -0.124 * ((v - E_Na) + 35.0) / (1.0 - exp(((v - E_Na) + 35.0) / 9.0)))
    push!(eqs, αₕ ~ 0.25 * exp(-((v - E_Na) + 90.0) / 12.0))
    push!(eqs, βₕ ~ 0.25 * (exp(((v - E_Na) + 62.0) / 6.0)) / exp(-((v - E_Na) + 90.0) / 12.0))
    push!(eqs, D(m) ~ αₘ * (1 - m) - βₘ * m)
    push!(eqs, D(h) ~ αₕ * (1 - h) - βₕ * h)
    push!(eqs, I_Na ~ g_Na * m^3 * h * (v - E_Na))

    # K gating
    push!(eqs, αₙ ~ 0.02 * ((v - E_K) - 25.0) / (1.0 - exp(-((v - E_K) - 25.0) / 9.0)))
    push!(eqs, βₙ ~ -0.002 * ((v - E_K) - 25.0) / (1.0 - exp(((v - E_K) - 25.0) / 9.0)))
    push!(eqs, D(n_gate) ~ αₙ * (1 - n_gate) - βₙ * n_gate)
    push!(eqs, I_K ~ g_K * n_gate^4 * (v - E_K))

    # Leak
    push!(eqs, I_L ~ g_L * (v - E_L))

    # Membrane equation: 'i' is acausal current, 'injector.I.u' is causal stimulus
    push!(eqs, C * D(v) ~ i + injector.I.u - I_Na - I_K - I_L)

    return extend(System(eqs, t, vars, params; systems=[injector, ground], name=name), oneport)
end

@component function VectorizedHHNeuron(; name, N::Int, C=1.0, g_Na=120.0, g_K=36.0, g_L=0.3, E_Na=50.0, E_K=-77.0,
E_L=-54.4, V_init=-65.0)
    # Parameters (scalars are automatically broadcasted by MTK if applied to arrays)
    @parameters C=C g_Na=g_Na g_K=g_K g_L=g_L E_Na=E_Na E_K=E_K E_L=E_L V_init=V_init
    params = SymbolicT[]
    push!(params, C, g_Na, g_K, g_L, E_Na, E_K, E_L, V_init)

    # Array Variables
    @variables begin
        V(t)[1:N] = fill(V_init, N)
        I_inj(t)[1:N] 
        m(t)[1:N] = zeros(Float64, N)
        h(t)[1:N] = ones(Float64, N)
        n_gate(t)[1:N] = zeros(Float64, N)
        I_Na(t)[1:N]
        I_K(t)[1:N]
        I_L(t)[1:N]
        αₘ(t)[1:N]
        βₘ(t)[1:N]
        αₕ(t)[1:N]
        βₕ(t)[1:N]
        αₙ(t)[1:N]
        βₙ(t)[1:N]
    end

    vars = SymbolicT[]
    push!(vars, V, I_inj, m, h, n_gate, I_Na, I_K, I_L, αₘ, βₘ, αₕ, βₕ, αₙ, βₙ)

    eqs = Equation[]

    # Na gating (using broadcasting .*)
    push!(eqs, αₘ ~ 0.182 .* (V .- E_Na .+ 35.0) ./ (1.0 .- exp.(-(V .- E_Na .+ 35.0) ./ 9.0)))
    push!(eqs, βₘ ~ -0.124 .* (V .- E_Na .+ 35.0) ./ (1.0 .- exp.((V .- E_Na .+ 35.0) ./ 9.0)))
    push!(eqs, αₕ ~ 0.25 .* exp.(-(V .- E_Na .+ 90.0) ./ 12.0))
    push!(eqs, βₕ ~ 0.25 .* (exp.((V .- E_Na .+ 62.0) ./ 6.0)) ./ exp.(-(V .- E_Na .+ 90.0) ./ 12.0))
    push!(eqs, D(m) ~ αₘ .* (1.0 .- m) .- βₘ .* m)
    push!(eqs, D(h) ~ αₕ .* (1.0 .- h) .- βₕ .* h)
    push!(eqs, I_Na ~ g_Na .* (m .^ 3) .* h .* (V .- E_Na))

    # K gating
    push!(eqs, αₙ ~ 0.02 .* (V .- E_K .- 25.0) ./ (1.0 .- exp.(-(V .- E_K .- 25.0) ./ 9.0)))
    push!(eqs, βₙ ~ -0.002 .* (V .- E_K .- 25.0) ./ (1.0 .- exp.((V .- E_K .- 25.0) ./ 9.0)))
    push!(eqs, D(n_gate) ~ αₙ .* (1.0 .- n_gate) .- βₙ .* n_gate)
    push!(eqs, I_K ~ g_K .* (n_gate .^ 4) .* (V .- E_K))

    # Leak
    push!(eqs, I_L ~ g_L .* (V .- E_L))

    # Membrane equation
    push!(eqs, D(V) ~ (I_inj .- I_Na .- I_K .- I_L) ./ C)

    return System(eqs, t, vars, params; systems=System[], name=name)
end


@component function STDPSynapse(; name, N::Int, W_init::Matrix{Float64}, A_plus=0.01, A_minus=0.01, tau_plus=20.0,
tau_minus=20.0, v_th=-20.0)
    @variables V_vec(t)[1:N] I_inj(t)[1:N] W(t)[1:N, 1:N]=W_init t_pre(t)[1:N]=fill(-1000.0, N) t_post(t)[1:N]=fill(-1000.0,
N)
    @parameters A_plus=A_plus A_minus=A_minus tau_plus=tau_plus tau_minus=tau_minus v_th_p=v_th

    # Synaptic conductance based on dynamic weight W
    eqs = Equation[]
    push!(eqs, I_inj ~ V_vec .* W)  # Simplified for example

    events = Any[]
    for j in 1:N # Pre-synaptic spikes
        root_eqs = [V_vec[j] ~ v_th_p]
        affect = [
            t_pre[j] ~ t,
            W[j, :] ~ clamp.(Pre(W[j, :]) .+ A_plus .* exp.(-(t .- Pre(t_post[:])) ./ tau_plus), 0.0, 1.0)
        ]
        push!(events, root_eqs => affect)
    end

    for i in 1:N # Post-synaptic spikes
        root_eqs = [V_vec[i] ~ v_th_p]
        affect = [
            t_post[i] ~ t,
            W[:, i] ~ clamp.(Pre(W[:, i]) .- A_minus .* exp.(-(t .- Pre(t_pre[:])) ./ tau_minus), 0.0, 1.0)
        ]
        push!(events, root_eqs => affect)
    end

    return System(eqs, t, [V_vec, I_inj, W, t_pre, t_post], [A_plus, A_minus, tau_plus, tau_minus, v_th_p]; continuous_events=events, name=name)
end
````

## File: src/BasicComponents.jl
````julia
"""
Soma Component: Represents a pure physical lipid bilayer membrane patch.
"""
@component function Capacitor(; name, C = 1.0, V_init = -65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    @parameters begin
        C = C
    end
    params = SymbolicT[]
    push!(params, C)
    
    @variables begin
        V(t) = V_init
    end
    vars = SymbolicT[]
    push!(vars, V)
    
    eqs = Equation[]
    push!(eqs, D(v) ~ i / C)
    push!(eqs, V ~ v)
    
    cap_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        name
    )
    return extend(cap_sys, oneport)
end



"""
CurrentSource Component: Converts a causal RealInput signal (u) 
into an acausal electrical current (i) injecting into a physical Node.
"""
@component function CurrentSource(; name)
    @named oneport = OnePort()
    @unpack i = oneport
    @named I = RealInput()
    
    vars = SymbolicT[]
    params = SymbolicT[]
    eqs = Equation[]
    push!(eqs, i ~ -I.u)
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    # We cast 'I' into a Vector{System} instead of leaving it as an untyped literal array
    subsystems = System[]
    push!(subsystems, I)
    
    source_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = subsystems, 
        initial_conditions, 
        guesses, 
        name
    )
    return extend(source_sys, oneport)
end

"""
fixed_reversal Component: A pure constant voltage source (Nernst battery).
"""
@component function FixedReversal(; name, E = 0.0)
    @named oneport = OnePort()
    @unpack v = oneport
    @parameters begin
        E = E
    end
    params = SymbolicT[]
    push!(params, E)
    vars = SymbolicT[]
    eqs = Equation[]
    push!(eqs, v ~ E)
    
    reversal_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        name
    )
    return extend(reversal_sys, oneport)
end

"""
SpikingCapacitor Component: Capacitor that automatically resets its voltage when a threshold is crossed 
"""
@component function SpikingCapacitor(; name, C = 10.0, V_th = -55.0, V_reset = -67.0, V_init = -65.0)
    @named oneport = OnePort()
    @unpack v, i = oneport
    
    @parameters begin
        C = C
        V_th = V_th
        V_reset = V_reset
    end
    params = params = SymbolicT[]
    push!(params, C, V_th, V_reset)
    
    @variables begin
        # Bind the incoming V_init default directly to the true differential state
        v(t) = V_init
        V(t)
    end
    # Include both v and V in the structural variables array
    vars = SymbolicT[]
    push!(vars, v, V)

    eqs = Equation[
        D(v) ~ i / C,
        V ~ v
    ]
    
    root_eqs = Equation[v ~ V_th]
    affect = Equation[v ~ V_reset]
    events = [root_eqs => affect]
    
    lif_sys = System(
        eqs, 
        t, 
        vars, 
        params; 
        systems = System[], 
        continuous_events = events,
        name
    )
    
    return extend(lif_sys, oneport)
end

@component function GapJunction(; name, R = 1.0)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport

    @parameters R = R
    params = SymbolicT[]
    push!(params, R)

    vars = SymbolicT[]

    eqs = Equation[]
    # The current flowing into port 1 is driven by the voltage difference.
    # By conservation of current, what goes into port 1 must come out of port 2.
    push!(eqs, i1 ~ (v1 - v2) / R)
    push!(eqs, i2 ~ -i1)

    return extend(System(eqs, t, vars, params; systems=System[], name=name), twoport)
end

@component function ChemicalSynapse(; name, g_max=2.0, τ=5.0, v_th=-20.0, w=0.5, E_rev=0.0)
    @named twoport = TwoPort()
    @unpack v1, i1, v2, i2 = twoport

    @parameters E_rev=E_rev g_max=g_max τ=τ v_th=v_th w=w
    params = SymbolicT[]
    push!(params, E_rev, g_max, τ, v_th, w)

    @variables s(t) = 0.0
    vars = SymbolicT[]
    push!(vars, s)

    eqs = Equation[]
    push!(eqs, i1 ~ 0.0)
    push!(eqs, D(s) ~ -s / τ)
    push!(eqs, i2 ~ (v2 - E_rev) * s * g_max)

    # Events should also be built cleanly
    root_eqs = Equation[]
    push!(root_eqs, v1 ~ v_th)
    affect = Equation[]
    push!(affect, s ~ Pre(s) + w)
    events = [root_eqs => affect]

    return extend(System(eqs, t, vars, params; systems=System[], continuous_events=events, name=name), twoport)
end

@component function AlphaSynapse(; name, g_max=3.0, τ=5.0, E_rev=0.0, v_th=-20.0, w=1.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev v_th=v_th w=w

    vars = SymbolicT[]
    push!(vars, s, I_syn, V_pre, V_post)

    params = SymbolicT[]
    push!(params, g_max, τ, E_rev, v_th, w)

    eqs = Equation[]
    push!(eqs, D(s) ~ -s / τ)
    push!(eqs, I_syn ~ (V_post - E_rev) * s * g_max)

    # Build event equations as explicitly typed Equation[] vectors
    root_eqs = Equation[]
    push!(root_eqs, V_pre ~ v_th)

    affect = Equation[]
    push!(affect, s ~ Pre(s) + w)
    push!(affect, V_pre ~ Pre(V_pre))   # Lock pre-synaptic voltage
    push!(affect, V_post ~ Pre(V_post)) # Lock post-synaptic voltage

    events = Any[] 
    push!(events, root_eqs => affect)

    # Explicitly pass systems=System[]
    return System(eqs, t, vars, params; systems=System[], continuous_events=events, name=name)
end


function spike_affect!(mod, obs, ctx, integ)
    j = ctx.j
    W = ctx.W
    N = ctx.N

    S_new = copy(mod.S)
    for i in 1:N
        S_new[j, i] += W[j, i]
    end
    return (; S = S_new)
end

@component function VectorizedAlphaSynapse(; name, N::Int, W::Matrix{Float64}, tau::Matrix{Float64}, g_max::Matrix{Float64},
E_rev=0.0, v_th=-20.0)
    # I_inj has NO default value, because it is an algebraic variable
    @variables V_vec(t)[1:N] I_inj(t)[1:N] S(t)[1:N, 1:N]=zeros(Float64, N, N)
    @parameters tau_p[1:N, 1:N]=tau g_max_p[1:N, 1:N]=g_max E_rev_p=E_rev v_th_p=v_th

    eqs = Equation[]
    push!(eqs, D(S) ~ -S ./ tau_p)

    for i in 1:N
        rhs = Num(0.0)
        for j in 1:N
            rhs += (V_vec[i] - E_rev_p) * S[j, i] * g_max_p[j, i]
        end
        push!(eqs, I_inj[i] ~ rhs)
    end

    events = Any[]
    for j in 1:N
        event = [V_vec[j] ~ v_th_p] => ImperativeAffect(
            spike_affect!,
            modified = (; S),
            observed = (;),
            ctx = (j=j, W=W, N=N)
        )
        push!(events, event)
    end

    # Type-stable SymbolicT[] vectors for fast precompilation
    vars = SymbolicT[]
    push!(vars, S)
    push!(vars, I_inj)
    push!(vars, V_vec)

    params = SymbolicT[]
    push!(params, tau_p)
    push!(params, g_max_p)
    push!(params, E_rev_p)
    push!(params, v_th_p)

    # No guesses needed, the system is perfectly balanced
    return System(eqs, t, vars, params;
                  continuous_events=events,
                  systems = System[],
                  name)
end
````

## File: src/connections.jl
````julia
"""
build_compartment: Constructs a single neural compartment (soma/dendrite).
If `stimulus_block` is provided, it drives the internal current injector.
If `open_injector=true`, the injector control input remains open for external wiring.
"""
function build_compartment(capacitor, channels; stimulus_block=nothing, name=:neuron)
    @named injector = CurrentSource()
    @named p = Pin()
    @named n = Pin()

    @variables V(t)  
    vars = SymbolicT[V]
    
    eqs = Equation[]
    
    # Connect capacitor to boundary pins
    push!(eqs, connect(capacitor.p, p))
    push!(eqs, connect(capacitor.n, n))
    push!(eqs, V ~ p.v) 
    
    # Connect all positive pins together
    p_connections = System[capacitor, injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # Connect all negative pins together
    n_connections = System[capacitor, injector]
    append!(n_connections, channels)
    push!(eqs, connect([sys.n for sys in n_connections]...))
    
    all_systems = System[p, n, capacitor, injector]
    append!(all_systems, channels)

    if stimulus_block !== nothing
        push!(eqs, connect(stimulus_block.output, injector.I))
        push!(all_systems, stimulus_block)
    end
    
    return System(eqs, t, vars, SymbolicT[]; systems = all_systems, name)
end

function build_synapse(gate, battery; name)
    @named pre_p  = Pin() # Pre-synaptic sensing active point
    @named pre_n  = Pin() # Pre-synaptic sensing reference point
    @named post_p = Pin() # Post-synaptic active injection point
    @named post_n = Pin() # Post-synaptic reference return point
    
    vars = SymbolicT[]
    params = SymbolicT[]
    initial_conditions = Dict{SymbolicT, SymbolicT}()
    guesses = Dict{SymbolicT, SymbolicT}()
    
    eqs = Equation[]
    # 1. Voltage sensing path (Pre-synaptic side)
    push!(eqs, connect(pre_p, gate.p1))
    push!(eqs, connect(pre_n, gate.n1))

    # 2. Current injection path (Post-synaptic side)
    push!(eqs, connect(post_p, gate.p2))
    push!(eqs, connect(gate.n2, battery.p))
    push!(eqs, connect(battery.n, post_n))
    
    subsystems = System[pre_p, pre_n, post_p, post_n, gate, battery]
    
    return System(eqs, t, vars, params; systems = subsystems, initial_conditions, guesses, name)
end

function build_vectorized_network(neurons::Vector{System}, synapse_blocks::Vector{System}; drivers=[], name=:vec_net)
    N = length(neurons)
    eqs = Equation[]
    all_systems = System[]
    append!(all_systems, neurons)

    # 1. Accumulate synaptic currents using Julia expressions
    I_exprs = [Num(0.0) for _ in 1:N]

    for block in synapse_blocks
        push!(all_systems, block)
        for i in 1:N
            push!(eqs, block.V_vec[i] ~ neurons[i].V)
            I_exprs[i] = I_exprs[i] + block.I_inj[i]
        end
    end

    # 2. Accumulate external stimulus directly into I_exprs
    for (target, stim) in drivers
        idx = target isa System ? findfirst(==(target), neurons) : target
        push!(all_systems, stim)
        I_exprs[idx] = I_exprs[idx] + stim.output.u
    end

    # 3. Map the final accumulated current directly to the injectors
    # (No if/else branches, just one clean equation per neuron)
    for i in 1:N
        push!(eqs, neurons[i].injector.I.u ~ I_exprs[i])
    end

    return System(eqs, t, SymbolicT[], SymbolicT[]; systems=all_systems, name=name)
end

# Helper to find the stim output for a specific neuron index
function stim_output(drivers, idx)
    for (target, stim) in drivers
        if (target isa System ? findfirst(==(target), neurons) : target) == idx
            return stim.output.u
        end
    end
    return Num(0.0)
end 


function build_fully_vectorized_network(neuron_block::System, synapse_blocks::Vector{System}; drivers=[], name=:vec_net)
    N = size(neuron_block.V)[1]

    eqs = Equation[]
    all_systems = System[neuron_block]
    append!(all_systems, synapse_blocks)

    # 1. Accumulate currents safely using a loop to maintain a mutable Vector{Num}
    I_exprs = [Num(0.0) for _ in 1:N]

    for block in synapse_blocks
        push!(eqs, block.V_vec ~ neuron_block.V)
        for i in 1:N
            I_exprs[i] = I_exprs[i] + block.I_inj[i]
        end
    end

    # 2. Add drivers (safe because I_exprs is still a standard Julia Vector)
    for (target, stim) in drivers
         @assert target isa Int "Fully vectorized networks require integer indices for drivers, because they contain a single monolithic neuron block."
        push!(all_systems, stim)
        I_exprs[target] = I_exprs[target] + stim.output.u
    end

    # 3. Push the connection equations
    for i in 1:N
        push!(eqs, neuron_block.I_inj[i] ~ I_exprs[i])
    end

    return System(eqs, t, SymbolicT[], SymbolicT[]; systems=all_systems, name=name)
end

"""
    build_electrical_network(compartments, axial_connections, synapse_connections; drivers=[], name=:network)

Construct an explicit, acausal circuit network from a flat list of compartments and two edge lists. 
Designed for multi-compartment spatial models, biophysical networks, and mixed-domain synapses.

# Arguments

- `compartments::Vector{System}`: A flat list of all compartment systems generated via `build_compartment`. 
  Indices in the connection lists refer to positions in this array.

- `axial_connections::Vector{<:Tuple}`: Internal structural connections (e.g., soma to dendrite). 
  Schema: `(pre_idx, post_idx, generator [, name::Symbol])`
  * `generator`: A function taking a keyword `name` that returns a `OnePort` or `TwoPort` system. 
    The builder will efficiently route standard `OnePort` components (like a `Resistor`) if no `p1` pin is found.

- `synapse_connections::Vector{<:Tuple}`: External synaptic connections between compartments.
  Schema: `(pre_idx, post_idx, generator [, name::Symbol])`
  * `generator`: A function taking a keyword `name` that returns a synapse system. 
    Synapses must be `TwoPort` systems for the acausal electrical current path. If the synapse 
    requires absolute voltage or calcium sensing, it should expose `RealInput`s named `V_pre_sense`, 
    `V_post_sense`, or `Ca_pre_sense`, which the builder will automatically wire.

# Keywords

- `drivers::Vector{Tuple{Int, System}}`: Optional list of causal input blocks targeting specific compartment indices. 
  Un-driven injectors are automatically grounded to `0.0` without instantiating redundant subsystems.
- `name::Symbol`: The system identifier for the resulting network.

# Example

```julia
compartments = [soma1, dend1, soma2]

# Efficient OnePort resistor for axial resistance
axial = [(1, 2, (; name) -> Resistor(R=0.5, name=name))]

# Mixed-domain synapse with absolute voltage sensing
synapses = [(2, 3, (; name) -> ChemicalSynapse(name=name, g_max=0.5))]

drivers = [(1, stim)] # Drive Soma1

net = build_electrical_network(compartments, axial, synapses; drivers=drivers)
```
"""

function build_electrical_network(compartments::Vector{System}, axial_connections, synapse_connections; drivers=[], name=:network)
N = length(compartments)
eqs = Equation[]
all_systems = System[]
append!(all_systems, compartments)

# 1. Single Global Ground
@named gnd = Ground()
push!(all_systems, gnd)
for i in 1:N
    push!(eqs, connect(compartments[i].n, gnd.g))
end

driven_compartments = Set{Int}()

# 2. Setup Driving Stimuli
for (target, stim) in drivers
    idx = target isa System ? findfirst(==(target), compartments) : target
    push!(driven_compartments, idx)
    push!(all_systems, stim)
    push!(eqs, connect(stim.output, compartments[idx].injector.I))
end

# 3. Axial Connections (Internal topology)
for conn in axial_connections
    pre_target, post_target, gen = conn[1], conn[2], conn[3]
    pre_idx = pre_target isa System ? findfirst(==(pre_target), compartments) : pre_target
    post_idx = post_target isa System ? findfirst(==(post_target), compartments) : post_target
    
    ax_name = length(conn) == 4 ? conn[4] : Symbol(:axial_, pre_idx, :_, post_idx)
    ax = gen(name=ax_name)
    push!(all_systems, ax)
    
    # Efficiently route OnePort vs TwoPort components
    if hasproperty(ax, :p1)
        push!(eqs, connect(compartments[pre_idx].p, ax.p1))
        push!(eqs, connect(compartments[post_idx].p, ax.p2))
        push!(eqs, connect(compartments[pre_idx].n, ax.n1))
        push!(eqs, connect(compartments[post_idx].n, ax.n2))
    else
        push!(eqs, connect(compartments[pre_idx].p, ax.p))
        push!(eqs, connect(compartments[post_idx].p, ax.n))
    end
end

# 4. Synaptic Connections (External topology)
for conn in synapse_connections
    pre_target, post_target, gen = conn[1], conn[2], conn[3]
    pre_idx = pre_target isa System ? findfirst(==(pre_target), compartments) : pre_target
    post_idx = post_target isa System ? findfirst(==(post_target), compartments) : post_target
    
    syn_name = length(conn) == 4 ? conn[4] : Symbol(:syn_, pre_idx, :_, post_idx)
    syn = gen(name=syn_name)
    push!(all_systems, syn)

    # Acausal wiring for synaptic currents
    push!(eqs, connect(compartments[pre_idx].p, syn.p1))
    push!(eqs, connect(compartments[post_idx].p, syn.p2))
    push!(eqs, connect(compartments[pre_idx].n, syn.n1))
    push!(eqs, connect(compartments[post_idx].n, syn.n2))

    # Auto-wiring for mixed-domain sensing
    if hasproperty(syn, :V_pre_sense)
        push!(eqs, syn.V_pre_sense.u ~ compartments[pre_idx].V)
    end
    if hasproperty(syn, :V_post_sense)
        push!(eqs, syn.V_post_sense.u ~ compartments[post_idx].V)
    end
    if hasproperty(syn, :Ca_pre_sense) && hasproperty(compartments[pre_idx], :Ca)
        push!(eqs, syn.Ca_pre_sense.u ~ compartments[pre_idx].Ca)
    end
end

# 5. Ground undriven injectors cleanly
for i in 1:N
    if !(i in driven_compartments)
        push!(eqs, compartments[i].injector.I.u ~ 0.0)
    end
end

return System(eqs, t, SymbolicT[], SymbolicT[]; systems = all_systems, name = name)
end

using Symbolics
using ModelingToolkit: unknowns, parameters, equations, @named, System, t_nounits as t

using ModelingToolkitStandardLibrary.Blocks: RealInput

using Symbolics: fixpoint_sub
using ModelingToolkit: unknowns, parameters, equations, @named, System, t_nounits as t, isparameter, is_derivative, getname

using ModelingToolkit: full_equations

struct Compartment
    sys::System
    interfaces::NamedTuple
end

function build_floating_compartment(capacitor, channels; name=:compartment)
    @named injector = CurrentSource()
    @named axial_injector = CurrentSource() 
    @named ground = Ground()

    @variables V(t)
    vars = SymbolicT[V]
    
    eqs = Equation[]
    
    push!(eqs, connect(capacitor.n, ground.g))
    push!(eqs, connect(injector.n, ground.g))
    push!(eqs, connect(axial_injector.n, ground.g))
    for c in channels
        push!(eqs, connect(c.n, ground.g))
    end
    
    p_connections = System[capacitor, injector, axial_injector]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))
    
    all_systems = System[ground, capacitor, injector, axial_injector]
    append!(all_systems, channels)

    push!(eqs, V ~ capacitor.v)
    
    sys = System(eqs, t, vars, SymbolicT[]; systems = all_systems, name)
    return Compartment(sys, (V=V, I_axial=axial_injector.I.u, I_ext=injector.I.u))
end



struct Cell
    sys::System
    compartments::Vector{Compartment}
    inputs::Vector{Any}  # <-- ADD THIS
end

function build_cell(compartments::Vector{Compartment}, axial_connections; drivers=[], ground_undriven=true, name=:cell)
    eqs = Equation[]
    all_systems = System[]
    driven_exts = Set{Int}()
    vars = SymbolicT[]
    cell_inputs = SymbolicT[]
    
    for (idx, comp) in enumerate(compartments)
        push!(all_systems, comp.sys)
    end
    
    for conn in axial_connections
        pre_idx, post_idx, R_val = conn
        V_pre = compartments[pre_idx].sys.V
        V_post = compartments[post_idx].sys.V
        I_ax_pre = compartments[pre_idx].sys.axial_injector.I.u
        I_ax_post = compartments[post_idx].sys.axial_injector.I.u
        
        I_flow = (V_pre - V_post) / R_val
        push!(eqs, I_ax_pre ~ -I_flow)
        push!(eqs, I_ax_post ~ I_flow)
    end
    
    for (target, stim) in drivers
        idx = target isa Int ? target : findfirst(==(target), compartments)
        push!(all_systems, stim)
        push!(eqs, compartments[idx].sys.injector.I.u ~ stim.output.u)
        push!(driven_exts, idx)
    end
    
    if ground_undriven
        for (idx, comp) in enumerate(compartments)
            if !(idx in driven_exts)
                push!(eqs, comp.sys.injector.I.u ~ 0.0)
            end
        end
    else
        # Network mode: directly expose the native RealInput of the CurrentSource
        for (idx, comp) in enumerate(compartments)
            if !(idx in driven_exts)
                push!(cell_inputs, comp.sys.injector.I.u)
            end
        end
    end
    
    @named cell_sys = System(eqs, t, vars, SymbolicT[]; systems=all_systems, inputs=cell_inputs, name=name)
    return Cell(cell_sys, compartments, cell_inputs)
end







struct Network
    sys::System
    nodes::DataFrame
    edges::DataFrame
    inputs::Vector{Any}
end

function build_network(cell::Cell, N::Int; synapse_connections=[], ground_inputs=true, name=:network)
    all_eqs = Equation[]
    all_vars_set = Set{SymbolicT}()
    all_params_set = Set{SymbolicT}()
    all_defaults = Dict{Any, Any}()
    all_systems = System[]
    all_events = []
    all_inputs = SymbolicT[]
    
    # Compile the cell with its specific external inputs
    compiled_cell = mtkcompile(cell.sys, inputs=cell.inputs)
    nodes = DataFrame(cell_idx=Int[], comp_idx=Int[], V=Any[], I_ext=Any[])
    
    # Updated to prefix parameters to avoid name clashes
    function make_new_name(sym, n_idx, is_param=false)
        sym_str = Base.replace(string(sym), "(t)" => "")
        flat_name = Base.replace(sym_str, "₊" => "_")
        prefix = is_param ? :p_ : :n_
        return Symbol(prefix, n_idx, :_, flat_name)
    end

    # Simple 1-level resolver for voltage aliases
    function resolve_v_var(compiled_sys, var)
        var_str = string(var)
        for u in unknowns(compiled_sys)
            if endswith(string(u), var_str)
                return u
            end
        end
        for eq in observed(compiled_sys)
            if eq isa Equation && endswith(string(eq.lhs), var_str)
                rhs_str = string(eq.rhs)
                for u in unknowns(compiled_sys)
                    if endswith(string(u), rhs_str)
                        return u
                    end
                end
            end
        end
        return nothing
    end

        println("\n--- DEBUG: Network Input Matching ---")
    println("Inputs in compiled_cell: ", ModelingToolkit.inputs(compiled_cell))
    for (c_idx, comp) in enumerate(cell.compartments)
        println("Searching for I_ext in comp $c_idx: '", string(comp.interfaces.I_ext), "'")
    end
    println("--------------------------------------\n")

    for n_idx in 1:N
        local_sub = Dict{Any, Any}()
        
        # Clone unknowns (this includes inputs)
        for u in unknowns(compiled_cell)
            new_name = make_new_name(u, n_idx, false)
            new_v = only(@variables $new_name(t))
            local_sub[u] = new_v
            push!(all_vars_set, new_v)
        end

        # Get string representations of all inputs to identify them
        input_strs = [Base.replace(string(inp), "(t)" => "") for inp in ModelingToolkit.inputs(compiled_cell)]
        
        # Clone parameters (with 'p_' prefix to avoid clashes)
        for p in parameters(compiled_cell)
            p_str = string(p)
            if any(endswith.(p_str, input_strs))
                continue
            end
            
            new_name = make_new_name(p, n_idx, true)
            new_p = only(@parameters $new_name)
            local_sub[p] = new_p
            push!(all_params_set, new_p)
        end

        # Map defaults
        for (orig_var, val) in ModelingToolkit.initial_conditions(compiled_cell)
            if haskey(local_sub, orig_var)
                all_defaults[local_sub[orig_var]] = val
            end
        end
        
        for eq in full_equations(compiled_cell)
            push!(all_eqs, fixpoint_sub(eq, local_sub))
        end
        
        for (c_idx, comp) in enumerate(cell.compartments)
            V_new = nothing
            I_ext_new = nothing
            
            # 1. Resolve Voltage
            true_V = resolve_v_var(compiled_cell, comp.interfaces.V)
            if true_V !== nothing
                V_new = local_sub[true_V]
            end
            
            # 2. Match the exact input using the compartment name
            comp_name = string(nameof(comp.sys))
            I_ext_str = "$(comp_name)₊injector₊I₊u(t)"
            
            for inp in ModelingToolkit.inputs(compiled_cell)
                if endswith(string(inp), I_ext_str)
                    I_ext_new = local_sub[inp]
                    break
                end
            end
            
            push!(nodes, (cell_idx=n_idx, comp_idx=c_idx, V=V_new, I_ext=I_ext_new))
            push!(all_inputs, I_ext_new)
        end




        for event in continuous_events(compiled_cell)
            if event isa Pair
                root_eqs = event.first
                affect = event.second
                new_root = [fixpoint_sub(eq, local_sub) for eq in root_eqs]
                
                if affect isa AbstractVector
                    new_affect = [fixpoint_sub(eq, local_sub) for eq in affect]
                    push!(all_events, new_root => new_affect)
                elseif affect isa ModelingToolkit.ImperativeAffect
                    new_mod = NamedTuple{keys(affect.modified)}([fixpoint_sub(v, local_sub) for v in affect.modified])
                    new_obs = NamedTuple{keys(affect.observed)}([fixpoint_sub(v, local_sub) for v in affect.observed])
                    new_affect = ModelingToolkit.ImperativeAffect(affect.f, new_mod, new_obs, affect.ctx)
                    push!(all_events, new_root => new_affect)
                end
            end
        end
    end

    syn_currents = Dict{Tuple{Int, Int}, Vector{Any}}()
    for (s_idx, conn) in enumerate(synapse_connections)
        pre_cell, pre_comp, post_cell, post_comp, gen = conn
        syn_name = Symbol(:syn_, s_idx)
        syn = gen(name=syn_name)
        push!(all_systems, syn)
        
        V_pre = nodes[(nodes.cell_idx .== pre_cell) .& (nodes.comp_idx .== pre_comp), :V][1]
        V_post = nodes[(nodes.cell_idx .== post_cell) .& (nodes.comp_idx .== post_comp), :V][1]
        
        push!(all_eqs, syn.V_pre ~ V_pre)
        push!(all_eqs, syn.V_post ~ V_post)
        
        key = (post_cell, post_comp)
        if !haskey(syn_currents, key)
            syn_currents[key] = Any[]
        end
        push!(syn_currents[key], syn.I_syn)
    end

    final_inputs = SymbolicT[]
    for row in eachrow(nodes)
        key = (row.cell_idx, row.comp_idx)
        I_ext = row.I_ext
        
        if haskey(syn_currents, key)
            I_syn_sum = sum(syn_currents[key])
            push!(all_eqs, I_ext ~ I_syn_sum)
        else
            if ground_inputs
                push!(all_eqs, I_ext ~ 0.0)
            else
                push!(final_inputs, I_ext)
            end
        end
    end
    
    @named net_sys = System(all_eqs, t, collect(all_vars_set), collect(all_params_set); 
                            initial_conditions=all_defaults, 
                            systems=all_systems, 
                            continuous_events=all_events, 
                            inputs=final_inputs, 
                            name=name)
                            
    return Network(net_sys, nodes, DataFrame(), final_inputs)
end
````

## File: src/MTKNeuralToolkit.jl
````julia
module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput, Constant, RealOutput, RealInputArray, RealOutputArray
import ModelingToolkitStandardLibrary.Electrical: Ground, OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, SymbolicT,ImperativeAffect
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
using DynamicQuantities
using DataFrames
import SymbolicUtils: scalarize
import Symbolics: Sym, Num


include("BasicComponents.jl")
export Ground, OnePort, Pin, Capacitor, SpikingCapacitor, CurrentSource, FixedReversal 
export ChemicalSynapse, GapJunction, VectorizedAlphaSynapse, AlphaSynapse

include("connections.jl")
export build_channel, build_compartment, build_floating_compartment, Cell, Compartment, build_cell, build_network



export build_synapse
export build_electrical_network, build_vectorized_network, build_fully_vectorized_network
# include("causal_connections.jl")
# export CausalSynapseGate, build_causal_synapse, VectorSynapsePopulation





include("tempgates.jl")
export GateSpec, GenericChannel
export nagates,lgates,kgates
export InlinedHHNeuron, VectorizedHHNeuron

include("loss_functions.jl")
export build_loss

end
````

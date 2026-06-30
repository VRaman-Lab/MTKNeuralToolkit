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
  deprecated.jl
  loss_functions.jl
  MTKNeuralToolkit.jl
  tempgates.jl
  vectorization.jl
```

# Files

## File: src/deprecated.jl
```julia
using Symbolics: fixpoint_sub, SymbolicT, Num, isarraysymbolic
using ModelingToolkit: unknowns, parameters, equations, @named, System, t_nounits as t, isparameter, is_derivative, getname, full_equations, continuous_events, observed, inputs, ImperativeAffect
using ModelingToolkitStandardLibrary.Blocks: RealInput
using ModelingToolkitStandardLibrary.Electrical: Ground
using DataFrames: DataFrame
using MacroTools: postwalk, @capture


function build_network(cell::Cell, N::Int; synapse_connections=[], ground_inputs=true, name=:network)

    compiled_cell = mtkcompile(cell.sys, inputs=cell.inputs)

    all_eqs = Equation[]
    all_vars = SymbolicT[]
    all_params = SymbolicT[]
    all_systems = System[]
    all_defaults = Dict{Any, Any}()
    all_events = []
    final_network_inputs = SymbolicT[]
    
    nodes = DataFrame(cell_idx=Int[], comp_idx=Int[], V=Any[], I_ext=Any[], Ca=Any[])
    
    driven_keys = Set{Tuple{Int,Int}}()
    for conn in synapse_connections
        post_cell, post_comp = conn[3], conn[4]
        push!(driven_keys, (post_cell, post_comp))
    end

    for n_idx in 1:N
        eqs, vars, ps, sub, defaults, events = clone_compiled_cell(compiled_cell, n_idx)
        append!(all_eqs, eqs)
        append!(all_vars, vars)
        append!(all_params, ps)
        merge!(all_defaults, defaults)
        append!(all_events, events)
        
        for (c_idx, comp) in enumerate(cell.compartments)
            V_orig = find_compiled_var(compiled_cell, comp.interfaces.V)
            I_ext_orig = find_compiled_var(compiled_cell, comp.interfaces.I_ext)
            
            V_new = sub[V_orig]
            I_ext_new = sub[I_ext_orig]
            
            all_defaults[V_new] = comp.V_init
            
            Ca_new = nothing
            if haskey(comp.interfaces, :Ca)
                Ca_orig = find_compiled_var(compiled_cell, comp.interfaces.Ca)
                Ca_new = sub[Ca_orig]
            end
            
            push!(nodes, (cell_idx=n_idx, comp_idx=c_idx, V=V_new, I_ext=I_ext_new, Ca=Ca_new))
            
            if !((n_idx, c_idx) in driven_keys)
                if ground_inputs
                    push!(all_eqs, I_ext_new ~ 0.0)
                else
                    push!(final_network_inputs, I_ext_new)
                end
            end
        end
    end

    syn_currents = Dict{Tuple{Int, Int}, Vector{Any}}()
    for (s_idx, conn) in enumerate(synapse_connections)
        pre_cell, pre_comp, post_cell, post_comp, gen = conn
        syn = gen(name=Symbol(:syn_, s_idx))
        push!(all_systems, syn)

        V_pre = nodes[(nodes.cell_idx .== pre_cell) .& (nodes.comp_idx .== pre_comp), :V][1]
        V_post = nodes[(nodes.cell_idx .== post_cell) .& (nodes.comp_idx .== post_comp), :V][1]

        push!(all_eqs, syn.V_pre ~ V_pre)
        push!(all_eqs, syn.V_post ~ V_post)

        if hasproperty(syn, :Ca_pre_sense)
            Ca_pre = nodes[(nodes.cell_idx .== pre_cell) .& (nodes.comp_idx .== pre_comp), :Ca][1]
            if Ca_pre !== nothing
                push!(all_eqs, syn.Ca_pre_sense.u ~ Ca_pre)
            end
        end

        key = (post_cell, post_comp)
        haskey(syn_currents, key) || (syn_currents[key] = Any[])
        
        if hasproperty(syn, :I_syn)
            push!(syn_currents[key], syn.I_syn)
        else
            push!(syn_currents[key], (syn.V_post - syn.E_rev) * syn.s * syn.g_max)
        end
    end

    for (key, currents) in syn_currents
        I_ext = nodes[(nodes.cell_idx .== key[1]) .& (nodes.comp_idx .== key[2]), :I_ext][1]
        push!(all_eqs, I_ext ~ sum(currents))
    end

    net_sys = System(all_eqs, t, all_vars, all_params;
                     initial_conditions=all_defaults,
                     systems=all_systems,
                     continuous_events=all_events,
                     inputs=final_network_inputs,
                     name=name)

    return Network(net_sys, nodes, DataFrame(), final_network_inputs)
end
```

## File: src/loss_functions.jl
```julia
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
```

## File: src/vectorization.jl
```julia
using Symbolics: SymbolicT, toexpr, parse_expr_to_symbolic, substitute
using ModelingToolkit: t_nounits as t, D_nounits as D, System, unknowns, parameters, defaults, Equation, getname
using MacroTools: postwalk, @capture, inexpr

# Set of operations that act on arrays as a whole, or are structural MTK components
const NO_BROADCAST_OPS = Set([
    :Differential, :D, :connect, :Pre, 
    :sum, :prod, :minimum, :maximum, :dot, :cross, 
    :length, :size, :eltype, :ndims, :axes, :eachindex, :stride,
    :colon, :(:), :reshape, :view, :getindex, :setindex!
])

"""
Helper function to inject broadcasting dots (`.`) into mathematical operations 
within a Julia `Expr` so it can act element-wise on Symbolic Arrays.
"""
function add_broadcasting(ex::Expr)
    postwalk(ex) do e
        if @capture(e, f_(xs__))
            should_bc = false
            
            if f isa Symbol
                should_bc = !(f in NO_BROADCAST_OPS)
            elseif f isa Expr
                if inexpr(f, :(Differential(_))) || inexpr(f, :D) || inexpr(f, :Pre)
                    should_bc = false
                else
                    should_bc = true
                end
            end

            if should_bc
                # Return surface AST for broadcasting: e.g. V .^ 3 becomes Expr(:call, :., :^, :V, 3)
                return Expr(:call, :., f, xs...)
            end
        end
        return e
    end
end

"""
    vectorize_system(scalar_sys::System, N::Int; scalar_params=Set{Symbol}())

Takes a scalar MTK system and returns a natively vectorized system of size N.
Parameters named in `scalar_params` are kept as scalars (e.g., shared constants).
Built strictly for precompilation type-stability.
"""
function vectorize_system(scalar_sys::System, N::Int; scalar_params=Set{Symbol}())
    sub = Dict{Any, Any}()
    
    # Precompilation-friendly typed vectors
    new_vars = SymbolicT[]
    new_params = SymbolicT[]
    new_eqs = Equation[]
    new_defaults = Dict{SymbolicT, Any}() # Any for values, since fill(v, N) is Vector{Float64}
    
    # 1. Map scalar unknowns to array unknowns
    for u in unknowns(scalar_sys)
        name = getname(u)
        u_arr = only(@variables $(name)(t)[1:N])
        push!(new_vars, u_arr)
        sub[u] = u_arr
    end
    
    # 2. Map scalar parameters to array parameters (or keep scalar)
    for p in parameters(scalar_sys)
        name = getname(p)
        if name in scalar_params
            p_new = only(@parameters $(name))
            push!(new_params, p_new)
            sub[p] = p_new
        else
            p_new = only(@parameters $(name)[1:N])
            push!(new_params, p_new)
            sub[p] = p_new
        end
    end
    
    # Build an expression-level substitution dictionary to bypass SymbolicUtils 
    # type-checking issues when promoting array powers (e.g. V .^ 3)
    expr_sub = Dict{Any, Any}()
    for (k, v) in sub
        expr_sub[toexpr(k)] = toexpr(v)
    end
    
    # 3. Transform equations
    for eq in equations(scalar_sys)
        # Convert to Julia Expr FIRST, before substitution
        expr_lhs = toexpr(eq.lhs)
        expr_rhs = toexpr(eq.rhs)
        
        # Inject broadcasting dots while everything is still scalar
        expr_lhs = add_broadcasting(expr_lhs)
        expr_rhs = add_broadcasting(expr_rhs)
        
        # Substitute scalar symbols with array symbols directly in the Expr AST
        expr_lhs_sub = postwalk(x -> haskey(expr_sub, x) ? expr_sub[x] : x, expr_lhs)
        expr_rhs_sub = postwalk(x -> haskey(expr_sub, x) ? expr_sub[x] : x, expr_rhs)
        
        expr_eq = :($expr_lhs_sub ~ $expr_rhs_sub)
        
        # parse_expr_to_symbolic avoids `eval` and world-age issues!
        new_eq = parse_expr_to_symbolic(expr_eq, @__MODULE__)
        push!(new_eqs, new_eq)
    end
    
    # 4. Handle defaults/initial conditions
    for (k, v) in defaults(scalar_sys) 
        if haskey(sub, k)
            # If scalar init was -65.0, array init is fill(-65.0, N)
            new_defaults[sub[k]] = fill(v, N) 
        end
    end

    return System(new_eqs, t, new_vars, new_params; 
                  defaults = new_defaults, 
                  systems = System[], # Explicitly typed Vector{System}
                  name = nameof(scalar_sys))
end
```

## File: src/tempgates.jl
```julia
using Symbolics: variable

struct GateSpec{I<:Integer, T<:AbstractFloat, F<:Function}
    name::Symbol
    power::I
    ic::T
    # A function taking voltage `v` and returning a tuple: (alpha_expr, beta_expr)
    dynamics::F 
end

@component function GenericChannel(; name, g, E_rev, gates::Vector{<:GateSpec}, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=N)
    end
    @unpack v, i = oneport
    
    @parameters g=g E_rev=E_rev
    vars = SymbolicT[]
    eqs = Equation[]
    init_conds = Dict{Any, Any}()
    
    if isempty(gates)
        # Pure leak channel (avoids broadcasting edge cases with empty gates)
        push!(eqs, i ~ g .* (v .- E_rev))
    else
        conductance_factor = true
        
        for gate in gates
            if isnothing(N)
                gate_var = only(@variables $(gate.name)(t))
                alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t))
                beta_var = only(@variables $(Symbol(gate.name, :_beta))(t))
                init_conds[gate_var] = gate.ic
            else
                gate_var = only(@variables $(gate.name)(t)[1:N])
                alpha_var = only(@variables $(Symbol(gate.name, :_alpha))(t)[1:N])
                beta_var = only(@variables $(Symbol(gate.name, :_beta))(t)[1:N])
                init_conds[gate_var] = fill(gate.ic, N)
            end
            
            push!(vars, gate_var, alpha_var, beta_var)
            
            alpha_expr, beta_expr = gate.dynamics(v)
            
            push!(eqs, alpha_var ~ alpha_expr)
            push!(eqs, beta_var ~ beta_expr)
            push!(eqs, D(gate_var) ~ alpha_expr .* (1.0 .- gate_var) .- beta_expr .* gate_var)
            
            conductance_factor = conductance_factor .* (gate_var .^ gate.power)
        end
        
        push!(eqs, i ~ g .* conductance_factor .* (v .- E_rev))
    end
    
    return extend(System(eqs, t, vars, [g, E_rev]; 
                       systems=System[], 
                       initial_conditions=init_conds, 
                       name=name), oneport)
end
```

## File: src/BasicComponents.jl
```julia
@component function Ground(; name, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named g = Pin()
        eqs = [g.v ~ 0]
    else
        @named g = VectorizedPin(N=N)
        eqs = [g.v ~ zeros(Float64, N)]
    end
    return System(eqs, t, SymbolicT[], SymbolicT[]; systems=[g], name=name)
end

@component function Capacitor(; name, C = 1.0, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named oneport = OnePort()
    else
        @named oneport = VectorizedOnePort(N=N)
    end
    @unpack v, i = oneport
    @parameters C=C
    # ./ works on both scalars and arrays natively in Symbolics
    eqs = Equation[D(v) ~ i ./ C]
    return extend(System(eqs, t, SymbolicT[], [C]; systems=System[], name=name), oneport)
end

@component function CurrentSource(; name, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named oneport = OnePort()
        @named I = RealInput()
    else
        @named oneport = VectorizedOnePort(N=N)
        @named I = RealInputArray(nin=N)
    end
    @unpack i = oneport
    
    eqs = Equation[i ~ -I.u]
    return extend(System(eqs, t, SymbolicT[], SymbolicT[]; systems=[I], name=name), oneport)
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

# ==========================================
# VECTORIZED ELECTRICAL COMPONENTS
# ==========================================

@connector function VectorizedPin(; name, N::Int, v = nothing, i = nothing)
    vars = @variables begin
        v(t)[1:N] = v
        i(t)[1:N] = i, [connect = Flow]
    end
    return System(Equation[], t, vars, SymbolicT[]; name=name)
end

@component function VectorizedOnePort(; name, N::Int, v = nothing, i = nothing)
    pars = @parameters begin
    end
    systems = @named begin
        p = VectorizedPin(N=N)
        n = VectorizedPin(N=N)
    end
    vars = @variables begin
        v(t)[1:N] = v
        i(t)[1:N] = i
    end
    equations = Equation[
        v ~ p.v - n.v,
        collect(p.i .+ n.i .~ 0.0)...,  # splat the collected equations
        i ~ p.i,
    ]

    return System(equations, t, vars, pars; name, systems)
end

@component function SynapsePort(; name, N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named p = Pin()
        @variables I_syn(t)
        vars = SymbolicT[I_syn]
        eqs = Equation[p.i ~ I_syn]
    else
        @named p = VectorizedPin(N=N)
        @variables I_syn(t)[1:N]
        vars = SymbolicT[I_syn]
        eqs = Equation[p.i ~ I_syn]
    end
    return System(eqs, t, vars, SymbolicT[]; systems=[p], name=name)
end


struct SynapseSpec
    pre_V::SymbolicT        # concrete variable from pre compartment
    post_I_syn::SymbolicT   # concrete variable from post compartment  
    post_V::SymbolicT       # for voltage-dependent synapses (NMDA)
    synapse::System         # the synapse component
end

function wire_synapse!(eqs, systems, spec::SynapseSpec)
    syn = spec.synapse
    push!(systems, syn)
    push!(eqs, syn.V_pre  ~ spec.pre_V)
    push!(eqs, syn.V_post ~ spec.post_V)
    # Accumulate — doesn't override, adds to whatever's already there
    push!(eqs, spec.post_I_syn ~ spec.post_I_syn + syn.I_syn)
end

@component function ExpSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    # Sigmoidal activation — smooth, no events needed
    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * (V_post - E_rev)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope]; 
                  systems=System[], name=name)
end

@component function AlphaSynapse(; name, g_max=1.0, τ=5.0, E_rev=0.0, V_th=-20.0, slope=2.0)
    @variables s1(t)=0.0 s2(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    
    eqs = [
        D(s1) ~ -s1 / τ + σ(V_pre - V_th),
        D(s2) ~ -s2 / τ + s1,           # cascaded low-pass → alpha shape
        I_syn ~ g_max * s2 * (V_post - E_rev)
    ]
    return System(eqs, t, [s1, s2, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, slope]; systems=System[], name=name)
end

@component function NMDASynapse(; name, g_max=1.0, τ=100.0, E_rev=0.0, V_th=-20.0, 
                                  Mg_conc=1.0, slope=2.0)
    @variables s(t)=0.0 I_syn(t) V_pre(t) V_post(t)
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th Mg_conc=Mg_conc slope=slope

    σ(x) = 1.0 / (1.0 + exp(-x/slope))
    # Mg block is a function of V_post — still fully causal
    mg_block(V) = 1.0 / (1.0 + Mg_conc * exp(-0.062 * V))
    
    eqs = [
        D(s) ~ -s / τ + σ(V_pre - V_th),
        I_syn ~ g_max * s * mg_block(V_post) * (V_post - E_rev)
    ]
    return System(eqs, t, [s, I_syn, V_pre, V_post], 
                  [g_max, τ, E_rev, V_th, Mg_conc, slope]; systems=System[], name=name)
end



@component function VectorizedExpSynapse(; name, N_pre, N_post, W,
                                            g_max=1.0, τ=5.0, E_rev=0.0,
                                            V_th=-20.0, slope=2.0)
    @variables s(t)[1:N_pre] I_syn(t)[1:N_post] V_pre(t)[1:N_pre] V_post(t)[1:N_post]
    @parameters g_max=g_max τ=τ E_rev=E_rev V_th=V_th slope=slope

    eqs = Equation[]

    # State dynamics: one ODE per pre-synaptic neuron
    for i in 1:N_pre
        push!(eqs, D(s[i]) ~ -s[i] / τ + 1.0 / (1.0 + exp(-(V_pre[i] - V_th) / slope)))
    end

    # Current output — only iterate non-zero W entries
    for j in 1:N_post
        nz_cols = findall(!iszero, @view W[j, :])
        if isempty(nz_cols)
            push!(eqs, I_syn[j] ~ 0.0)
        else
            synaptic_drive = sum(W[j, i] * s[i] for i in nz_cols)
            push!(eqs, I_syn[j] ~ g_max * (V_post[j] - E_rev) * synaptic_drive)
        end
    end

    return System(eqs, t, [s, I_syn, V_pre, V_post], [g_max, τ, E_rev, V_th, slope];
                  systems=System[], name=name)
end
```

## File: src/connections.jl
```julia
# connections.jl

using Symbolics: SymbolicT
using ModelingToolkit: t_nounits as t, connect, Equation, System, @named, getproperty, nameof

# =========================================================
# 1. STRUCT DEFINITIONS
# =========================================================

struct Compartment
    sys::System
    interfaces::NamedTuple
    V_init::Float64
end

struct Cell
    sys::System
    compartments::Vector{Compartment}
    inputs::Vector{Any}
end

struct Network
    sys::System
    nodes::DataFrame
    edges::DataFrame
    inputs::Vector{Any}
end

struct SynapseSpec
    pre_V::SymbolicT
    post_V::SymbolicT
    post_I_syn::SymbolicT
    synapse::System
end

# =========================================================
# 2. COMPARTMENT & CELL BUILDERS
# =========================================================

function build_compartment(capacitor, channels; name=:compartment, V_init=-65.0, 
                           N::Union{Int, Nothing}=nothing)
    if isnothing(N)
        @named injector  = CurrentSource()
        @named syn_port  = SynapsePort()
        @named p = Pin()
        @named n = Pin()
        init_v = V_init
    else
        @named injector  = CurrentSource(N=N)
        @named syn_port  = SynapsePort(N=N)
        @named p = VectorizedPin(N=N)
        @named n = VectorizedPin(N=N)
        init_v = fill(V_init, N)
    end

    vars = SymbolicT[]
    eqs  = Equation[]

    # 1. Connect all negative terminals together
    n_pins = Any[capacitor.n, injector.n, n]
    for c in channels
        push!(n_pins, c.n)
    end
    push!(eqs, connect(n_pins...))

    # 2. Connect all positive terminals together (incl. syn_port)
    p_connections = System[capacitor, injector, syn_port]
    append!(p_connections, channels)
    push!(eqs, connect([sys.p for sys in p_connections]...))

    # 3. Expose boundary pin for acausal connections (gap junctions)
    push!(eqs, connect(p, capacitor.p))

    all_systems = System[capacitor, injector, syn_port, p, n]
    append!(all_systems, channels)

    sys = System(eqs, t, vars, SymbolicT[];
                 systems = all_systems,
                 initial_conditions = Dict(capacitor.v => init_v),
                 name)

    cap_name = nameof(capacitor)
    V_state  = getproperty(sys, cap_name).v

    interfaces = (
        V       = V_state,
        p_pin   = getproperty(sys, nameof(p)),
        n_pin   = getproperty(sys, nameof(n)),
        I_ext   = getproperty(sys, nameof(injector)).I.u,
        I_syn   = getproperty(sys, nameof(syn_port)).I_syn,
        cap_name = cap_name
    )
    return Compartment(sys, interfaces, V_init)
end

function build_cell(compartments::Vector{Compartment}, axial_connections;
                    drivers=[], ground_undriven=true, name=:cell)
    eqs = Equation[]
    all_systems = System[]
    driven_exts = Set{Int}()
    gap_junctioned = Set{Int}()
    vars = SymbolicT[]
    cell_inputs = SymbolicT[]

    for comp in compartments
        push!(all_systems, comp.sys)
    end

    # 1. Connect all n pins together
    n_pins = [comp.interfaces.n_pin for comp in compartments]
    if length(n_pins) > 1
        push!(eqs, connect(n_pins...))
    end

    # 2. Global ground
    @named ground = Ground()
    push!(all_systems, ground)
    push!(eqs, connect(ground.g, compartments[1].interfaces.n_pin))

    # 3. Axial connections via GapJunction (replaces axial_injector)
    for (i, conn) in enumerate(axial_connections)
        pre_idx, post_idx, R_val = conn
        gj = GapJunction(R=R_val, name=Symbol(:gj_, i))
        push!(all_systems, gj)

        # Connect gap junction between the two compartments
        push!(eqs, connect(compartments[pre_idx].interfaces.p_pin, gj.p1))
        push!(eqs, connect(gj.n1, compartments[pre_idx].interfaces.n_pin))
        push!(eqs, connect(compartments[post_idx].interfaces.p_pin, gj.p2))
        push!(eqs, connect(gj.n2, compartments[post_idx].interfaces.n_pin))

        push!(gap_junctioned, pre_idx, post_idx)
    end

    # 4. Drivers
    for (target, stim) in drivers
        idx = target isa Int ? target : findfirst(==(target), compartments)
        push!(driven_exts, idx)

        if stim isa System
            push!(all_systems, stim)
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim.output.u)
        elseif stim isa Number
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim)
        elseif stim isa AbstractVector
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim)
        end
    end

    # 5. Ground undriven I_ext
    if ground_undriven
        for (idx, comp) in enumerate(compartments)
            if !(idx in driven_exts)
                push!(eqs, comp.interfaces.I_ext ~ 0.0)
            end
        end
    else
        for (idx, comp) in enumerate(compartments)
            if !(idx in driven_exts)
                push!(cell_inputs, comp.interfaces.I_ext)
            end
        end
    end

    # 6. Ground I_syn (no synapses at cell level)
    for comp in compartments
        push!(eqs, comp.interfaces.I_syn ~ 0.0)
    end

    # 7. Ground p_pin for non-gap-junctioned compartments
    for (idx, comp) in enumerate(compartments)
        if !(idx in gap_junctioned)
            push!(eqs, comp.interfaces.p_pin.i ~ 0.0)
        end
    end

    @named cell_sys = System(eqs, t, vars, SymbolicT[];
                             systems=all_systems,
                             inputs=cell_inputs,
                             name=name)
    return Cell(cell_sys, compartments, cell_inputs)
end

# =========================================================
# 3. SYNAPSE WIRING
# =========================================================

"""
    wire_synapses!(eqs, systems, specs)

Wires a collection of SynapseSpecs into the network equations.
Pre-collects convergent synapses by target and writes one sum equation per target.
Returns the set of driven I_syn targets (for grounding the rest).
"""
function wire_synapses!(eqs::Vector{Equation}, systems::Vector{System},
                        specs::Vector{SynapseSpec})
    syn_by_target = Dict{SymbolicT, Vector{SymbolicT}}()

    for spec in specs
        push!(systems, spec.synapse)
        push!(eqs, spec.synapse.V_pre  ~ spec.pre_V)
        push!(eqs, spec.synapse.V_post ~ spec.post_V)

        if spec.post_I_syn isa AbstractArray
            # Block synapse: expand array to individual elements
            for i in 1:length(spec.post_I_syn)
                key = spec.post_I_syn[i]
                haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
                push!(syn_by_target[key], spec.synapse.I_syn[i])
            end
        else
            # Scalar synapse
            key = spec.post_I_syn
            haskey(syn_by_target, key) || (syn_by_target[key] = SymbolicT[])
            push!(syn_by_target[key], spec.synapse.I_syn)
        end
    end

    for (target, currents) in syn_by_target
        push!(eqs, target ~ sum(currents))
    end

    return Set{SymbolicT}(keys(syn_by_target))
end



# =========================================================
# 4. NETWORK BUILDER
# =========================================================

function build_acausal_network(compartments::Vector{Compartment};
                                gap_junctions=[],
                                synapse_specs=SynapseSpec[],
                                drivers=[],
                                name=:network,
                                N::Union{Int, Nothing}=nothing)
    num_compartments = length(compartments)
    eqs = Equation[]
    all_systems = System[]

    for comp in compartments
        push!(all_systems, comp.sys)
    end

    # 1. Tie all grounds together
    n_pins = [compartments[i].interfaces.n_pin for i in 1:num_compartments]
    if length(n_pins) > 1
        push!(eqs, connect(n_pins...))
    end

    if isnothing(N)
        @named gnd = Ground()
    else
        @named gnd = Ground(N=N)
    end
    push!(all_systems, gnd)
    push!(eqs, connect(gnd.g, compartments[1].interfaces.n_pin))

    driven_compartments = Set{Int}()
    gap_junctioned = Set{Int}()

    # 2. Driving stimuli
    for (target, stim) in drivers
        idx = target isa Compartment ? findfirst(==(target), compartments) : target
        push!(driven_compartments, idx)

        if stim isa System
            push!(all_systems, stim)
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim.output.u)
        elseif stim isa AbstractVector
            push!(eqs, compartments[idx].interfaces.I_ext ~ stim)
        elseif stim isa Number
            if isnothing(N)
                push!(eqs, compartments[idx].interfaces.I_ext ~ stim)
            else
                push!(eqs, compartments[idx].interfaces.I_ext ~ fill(stim, N))
            end
        end
    end

    # 3. Ground undriven I_ext
    for i in 1:num_compartments
        if !(i in driven_compartments)
            if isnothing(N)
                push!(eqs, compartments[i].interfaces.I_ext ~ 0.0)
            else
                push!(eqs, compartments[i].interfaces.I_ext ~ zeros(Float64, N))
            end
        end
    end

    # 4. Wire gap junctions via p_pin
    for (i, gj_spec) in enumerate(gap_junctions)
        comp_i, comp_j, R = gj_spec
        gj = GapJunction(R=R, name=Symbol(:gj_, i))
        push!(all_systems, gj)

        push!(eqs, connect(compartments[comp_i].interfaces.p_pin, gj.p1))
        push!(eqs, connect(gj.n1, compartments[comp_i].interfaces.n_pin))
        push!(eqs, connect(compartments[comp_j].interfaces.p_pin, gj.p2))
        push!(eqs, connect(gj.n2, compartments[comp_j].interfaces.n_pin))

        push!(gap_junctioned, comp_i, comp_j)
    end

    # 5. Wire synapses (pre-collects by target, handles convergence)
    driven_syn_targets = wire_synapses!(eqs, all_systems, synapse_specs)

    # 6. Ground non-synapsed I_syn
    for comp in compartments
        if isnothing(N)
            if !(comp.interfaces.I_syn in driven_syn_targets)
                push!(eqs, comp.interfaces.I_syn ~ 0.0)
            end
        else
            for i in 1:N
                i_syn_i = comp.interfaces.I_syn[i]
                if !(i_syn_i in driven_syn_targets)
                    push!(eqs, i_syn_i ~ 0.0)
                end
            end
        end
    end

    # 7. Ground non-gap-junctioned p_pin.i
    for i in 1:num_compartments
        if !(i in gap_junctioned)
            if isnothing(N)
                push!(eqs, compartments[i].interfaces.p_pin.i ~ 0.0)
            else
                push!(eqs, compartments[i].interfaces.p_pin.i ~ zeros(Float64, N))
            end
        end
    end

    net_sys = System(eqs, t, SymbolicT[], SymbolicT[];
                     systems = all_systems, name = name)
    return Network(net_sys, DataFrame(), DataFrame(), SymbolicT[])
end


function build_synapse_block(pre_comp, post_comp, W; name, 
                             synapse_type=VectorizedExpSynapse, kwargs...)
    N_pre  = size(W, 2)
    N_post = size(W, 1)
    syn = synapse_type(N_pre=N_pre, N_post=N_post, W=W; name=name, kwargs...)
    return SynapseSpec(pre_comp.interfaces.V, post_comp.interfaces.V,
                       post_comp.interfaces.I_syn, syn)
end
```

## File: src/MTKNeuralToolkit.jl
```julia
module MTKNeuralToolkit

using ModelingToolkit
import ModelingToolkitStandardLibrary.Blocks: RealInput, Constant, RealOutput, RealInputArray, RealOutputArray
import ModelingToolkitStandardLibrary.Electrical: OnePort, TwoPort, Pin
using ModelingToolkit: t_nounits as t, D_nounits as D, connect, SymbolicT, ImperativeAffect
using ModelingToolkit: mtkcompile, Pre
using OrdinaryDiffEq
using DynamicQuantities
using DataFrames
import SymbolicUtils: scalarize
import Symbolics: Sym, Num

include("BasicComponents.jl")
export Ground, OnePort, Pin, Capacitor, SpikingCapacitor, CurrentSource, FixedReversal 
export ChemicalSynapse, GapJunction, AlphaSynapse, SynapseSpec

export VectorizedPin, VectorizedOnePort
export GenericChannel

include("connections.jl")
export build_compartment, Cell, Compartment, build_cell, build_network
export build_synapse
export build_acausal_network, build_synapse_block

include("tempgates.jl")
export GateSpec, GenericChannel

export ExpSynapse, VectorizedExpSynapse

include("vectorization.jl")
export vectorize_system

include("loss_functions.jl")
export build_loss

end
```

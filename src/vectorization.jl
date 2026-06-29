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

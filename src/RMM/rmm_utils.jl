function lti_min_max_norm(lti::Vector)
    lti_min, lti_max = extrema(lti)
    range = lti_max - lti_min + 123456e-12
    
    return [2 * (x - lti_min) / range - 1 for x in lti]
end

#=function lti_min_max_norm(lti::Float64)
    if lti>0
        return sqrt(lti)
    end
    return -(sqrt(-lti))
end=#


function make_lti_vecs(τ; δ=0.01)
    # Handle both input formats - flatten if nested
    τ_flat = τ isa Vector{Vector{Float64}} ? vcat(τ...) : τ
    
    # Compute eigenvalues using zero-order-hold transformation  
    λ = exp.(-δ ./ τ_flat)
    
    A_Mat = λ  
    B_Vec = 1.0 .- λ 
    
    return A_Mat, B_Vec
end

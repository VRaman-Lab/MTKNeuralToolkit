abstract type AbstractGeometry end
struct NoGeometry <: AbstractGeometry end

Base.@kwdef struct Geometry <: AbstractGeometry
    area::Float64 = 0.0628   # default to common STG area in cm^2
    C_m::Float64  = 1.0      # default to standard specific capacitance in uF/cm^2
end

# 2. Multiple Dispatch Rules for Biophysics
# Capacitance extraction
get_capacitance(C, geom::NoGeometry) = C
get_capacitance(C, geom::Geometry) = geom.C_m * geom.area

# Conductance extraction
get_conductance(g, geom::NoGeometry) = g
get_conductance(g, geom::Geometry) = g * geom.area

# Calcium conversion factor extraction 
get_ca_conversion_factor(conv, geom::NoGeometry, tauCa) = conv
get_ca_conversion_factor(conv, geom::Geometry, tauCa) = 0.94 / (geom.C_m * geom.area * tauCa)

get_synaptic_conductance(g, geom::NoGeometry) = g
get_synaptic_conductance(g, geom::Geometry) = g * geom.area



abstract type AbstractMorphology end
struct NoMorphology <: AbstractMorphology end

 Base.@kwdef struct Morphology <: AbstractMorphology
    position::Tuple{Float64, Float64, Float64} = (0.0, 0.0, 0.0) # x, y, z in space
    shape::Symbol = :spherical                              # :spherical, :cylindrical, :point
    dimensions::NamedTuple = (radius=10.0,)                 # in microns
    color::Symbol = :blue                                   # rendering hint
end

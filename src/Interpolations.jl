module Interpolations

using LinearAlgebra
using Statistics
using Base.Threads
using ..Meshfree4ScalarEq.ParticleGrids
using ..Meshfree4ScalarEq.SimSettings
using ..Meshfree4ScalarEq.HyperbolicPDEs
using ..Meshfree4ScalarEq.FluxFunctions
using ..Meshfree4ScalarEq.MLSWeightFunctions
using ..Meshfree4ScalarEq.MOOD
using ..InterpolationUtils

export functionInterpolation!, gradInterpolation!, setCurvatures!, initTimeStep, getStencil, LaxFriedrichsGradient,
     initGI!, initGIBuffers!


include("InterpolationUtils.jl")

function initGI!(::NoFallbackGrad, kwargs...)
    return
end

function initGIBuffers!(::NoFallbackGrad, kwargs...)
    return
end

function initTimeStep(g::GradientInterpolator, pg::ParticleGrid) end  # Function called at the start of a time step (order RK-stage)
"""
Ensures a vector `v` has at least capacity `n`.
Resizes if `length(v) < n`.
"""
function _ensure_capacity!(v::AbstractVector, n::Int)
    if length(v) < n
        n = n + n ÷ 4
        resize!(v, n)
    end
    return nothing
end

"""
    sortFlux(flux_ij::Real, flux_ji::Real, deltaX::Real)::Tuple{<:Real, <:Real}

Given a reconstruction of the state at the midpoint from the cell center flux1, and a state reconstruction from the neighbouring point, return the left and right state based on the relative orientation of the points.
"""
function sortFlux(flux_ij::Float64, flux_ji::Float64, deltaX::Float64)::Tuple{Float64, Float64}
    if deltaX > 0.0
        return (flux_ij, flux_ji)  # left state, right state
    else
        return (flux_ji, flux_ij)
    end
end

"""
    sortFlux(flux_ij::Real, flux_ji::Real, deltaX::Real)::Tuple{<:Real, <:Real}

Given a reconstruction of the state at the midpoint from the cell center flux1, and a state reconstruction from the neighbouring point, return the left and right state in x and y direction.
"""
function sortFlux(flux_ij::Float64, flux_ji::Float64, deltaX::Float64, deltaY::Float64)::Tuple{Float64, Float64, Float64, Float64}
    if deltaX > 0.0 && deltaY > 0.0
        return (flux_ij, flux_ji, flux_ij, flux_ji)
    elseif deltaX > 0.0 && deltaY < 0.0 
        return (flux_ij, flux_ji, flux_ji, flux_ij)
    elseif deltaX < 0.0 && deltaY > 0.0
        return (flux_ji, flux_ij, flux_ij, flux_ji)
    else
        return (flux_ji, flux_ij, flux_ji, flux_ij)
    end
end

include("./CentralGradient.jl")
include("./MUSCL.jl")
include("./Upwind.jl")
include("./WENO.jl")

end # End Module Interpolations
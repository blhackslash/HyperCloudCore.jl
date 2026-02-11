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

export functionInterpolation!, gradInterpolation!, setCurvatures!, GradientInterpolator, initTimeStep, UpwindGradient, CentralGradient, WENO, MUSCL, AxelMUSCL, DumbserWENO, getStencil, LaxFriedrichsGradient, MUSCLlimited,
    NoFallbackGrad, initGI!, initGIBuffers!


"""
    GradientInterpolator

In case of unstructured grids, the spatial gradient is approximated using a moving least squares (MLS) method based on Taylor polynomials.
These algorithms are implemented as follows. Each method is a struct that is a subtype of GradientInterpolator. The gradient 
at a gridpoint can then be computed using the ()-operator; see for example UpwindGradient and CentralGradient. These objects select
the correct stencil and then call the MLS routine (gradInterpolation).
"""
abstract type GradientInterpolator end

# Fallback Gradient interpolator for no fallback
struct NoFallbackGrad <: GradientInterpolator end

function initGI!(::NoFallbackGrad, kwargs...)
    return
end

function initGIBuffers!(::NoFallbackGrad, kwargs...)
    return
end

function initTimeStep(g::GradientInterpolator, particleGrid::ParticleGrid) end  # Function called at the start of a time step (order RK-stage)
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

# # ------------------------------- Dumbser WENO -------------------------------

# function getStencil(deltaX::Real, deltaY::Real, s::Int64)
#     stencil = convert(Int64, div(s*(atan(deltaY, deltaX) + pi)*4/pi, s))
#     stencil = stencil == 8 ? 0 : stencil  # Negative x-axis should be contained in stencil 0
#     return stencil
# end

# struct DumbserWENO <: GradientInterpolator
#     order::Int64
#     res::AbstractVector{Float64}
#     weightFunction::MLSWeightFunction
#     s::Integer  # amount of one-sided stencils
#     gradients::Matrix{Float64}
#     weights::AbstractVector{Float64}

#     function DumbserWENO(order::Int64 = 2; weightFunction::MLSWeightFunction = exponentialWeightFunction())
#         @assert order == 2 "Order must be to two, since the WENO weights require a second derivative."
#         new(order, AbstractVector{Float64}(undef, 5), weightFunction, 8, Matrix{Float64}(undef, (5, 9)), AbstractVector{Float64}(undef, 9))
#     end
# end

# function (weno::DumbserWENO)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::AbstractVector{<:Real}, eq::LinearAdvection{2}, settings::SimSetting; setCurvature::Bool=true)::Real
#     @assert settings.interpRange >= sqrt(5.0^2 + 3.0^2)*particleGrid.dx "Interpolation must be sufficiently larger, otherwise one cannot guarantee sufficient neighbours are found." 
#     particle = particleGrid.grid[particleIndex]
#     Npts = length(particle.neighbourIndices)

#     # Divide points in stencils
#     windowMatrix = zeros(Bool, (Npts, weno.s+1))
#     windowMatrix[:, 1] .= true  # First column is the central stencil

#     for i in eachindex(particle.neighbourIndices)
#         nbIndex = particle.neighbourIndices[i]
#         deltaX, deltaY = getDistance(particleGrid, particleIndex, nbIndex)
#         particle.dxVec[i] = deltaX/settings.interpRange
#         particle.dyVec[i] = deltaY/settings.interpRange
#         particle.dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
#         stencil = getStencil(deltaX, deltaY, weno.s)  # in [0, 7]
#         windowMatrix[i, stencil+2] = true
#     end
#     for stencil in 1:weno.s+1
#         particle.wVec .= weno.weightFunction(particle.dxVec, particle.dyVec; param=settings.interpAlpha, normalisation=1.0)
        
#         # There should be at least 5 points in each stencil!
#         @assert count(windowMatrix[:, stencil]) >= 5 "($(particle.pos[1]), $(particle.pos[2])), $(count(windowMatrix[:, stencil])), $(stencil)"
#         gradInterpolation!(particle.dxVec[windowMatrix[:, stencil]], particle.dyVec[windowMatrix[:, stencil]], particle.wVec[windowMatrix[:, stencil]], particle.dfVec[windowMatrix[:, stencil]], weno.res; order=weno.order)

#         # Rescale results
#         weno.gradients[1, stencil] = weno.res[1]/settings.interpRange
#         weno.gradients[2, stencil] = weno.res[2]/settings.interpRange  
#         weno.gradients[3, stencil] = weno.res[3]/(settings.interpRange^2)
#         weno.gradients[4, stencil] = weno.res[4]/(settings.interpRange^2)
#         weno.gradients[5, stencil] = weno.res[5]/(settings.interpRange^2)

#         # Compute weights
#         r = 4
#         eps = 1e-14
#         lambda = (stencil == 1) ? 10^5 : 1.0
#         weno.weights[stencil] = lambda/((eps + sum((x^2 for x in weno.gradients[:, stencil])))^r)
#     end

#     # Normalise weights
#     weno.weights .= weno.weights ./ sum(weno.weights)
    
#     if setCurvature 
#         particle.curvature[1] = 0.0
#         particle.curvature[2] = 0.0
#         for i in eachindex(weno.weights)  # Write out inner product
#             particle.curvature[1] += weno.weights[i]*weno.gradients[3, i]
#             particle.curvature[2] += weno.weights[i]*weno.gradients[4, i]
#         end
#     end

#     # Compute divergence
#     res = 0.0
    
#     for i in eachindex(weno.weights)  # Write out inner product
#         res += weno.weights[i]*(weno.gradients[1, i]*eq.vel[1] + eq.vel[2]*weno.gradients[2, i])
#     end
#     return res
# end

include("./CentralGradient.jl")
include("./MUSCL.jl")
include("./Upwind.jl")
include("./WENO.jl")

end # End Module Interpolations
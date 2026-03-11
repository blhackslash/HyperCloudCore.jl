module FluxFunctions

using ..Meshfree4ScalarEq.HyperbolicPDEs

export NumericalFluxFunction, RusanovFlux, UpwindFlux, RoeDiffusiveFlux

abstract type NumericalFluxFunction end;

# ---------- RusanovFlux (LLF)
struct RusanovFlux <: NumericalFluxFunction end

# 1D or generic fallback
@inline function (rusanov::RusanovFlux)(leftState::Float64, rightState::Float64, eq::ScalarHyperbolicPDE)
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    s = max(abs(velocity(eq, leftState)), abs(velocity(eq, rightState)))
    return 0.5 * (leftFlux + rightFlux - s * (rightState - leftState))
end

# 2D Optimized simultaneous evaluation
@inline function (rusanov::RusanovFlux)(fmx::Float64, fpx::Float64, fmy::Float64, fpy::Float64, eq::ScalarHyperbolicPDE{2})
    # Evaluate and extract ONLY the X components
    fx_l = flux(eq, fmx)[1]
    fx_r = flux(eq, fpx)[1] 
    vx_l = velocity(eq, fmx)[1]
    vx_r = velocity(eq, fpx)[1]
    sx = max(abs(vx_l), abs(vx_r))
    num_fx = 0.5 * (fx_l + fx_r - sx * (fpx - fmx))
    
    # Evaluate and extract ONLY the Y components
    fy_l = flux(eq, fmy)[2]
    fy_r = flux(eq, fpy)[2]
    vy_l = velocity(eq, fmy)[2]
    vy_r = velocity(eq, fpy)[2]
    sy = max(abs(vy_l), abs(vy_r))
    num_fy = 0.5 * (fy_l + fy_r - sy * (fpy - fmy))
    
    return num_fx, num_fy
end

# ---------- Upwind Flux
struct UpwindFlux <: NumericalFluxFunction end

# 1D or generic fallback
@inline function (upwind::UpwindFlux)(leftState::Float64, rightState::Float64, eq::ScalarHyperbolicPDE) 
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    a = leftState == rightState ? velocity(eq, leftState) : (leftFlux - rightFlux) / (leftState - rightState)
    return 0.5 * (leftFlux + rightFlux - abs(a) * (rightState - leftState))
end

# 2D Optimized simultaneous evaluation
@inline function (upwind::UpwindFlux)(fmx::Float64, fpx::Float64, fmy::Float64, fpy::Float64, eq::ScalarHyperbolicPDE{2}) 
    # X-direction
    fx_l = flux(eq, fmx)[1]
    fx_r = flux(eq, fpx)[1]
    vx_l = velocity(eq, fmx)[1]
    ax = fmx == fpx ? vx_l : (fx_l - fx_r) / (fmx - fpx)
    num_fx = 0.5 * (fx_l + fx_r - abs(ax) * (fpx - fmx))
    
    # Y-direction
    fy_l = flux(eq, fmy)[2]
    fy_r = flux(eq, fpy)[2]
    vy_l = velocity(eq, fmy)[2]
    ay = fmy == fpy ? vy_l : (fy_l - fy_r) / (fmy - fpy)
    num_fy = 0.5 * (fy_l + fy_r - abs(ay) * (fpy - fmy))
    
    return num_fx, num_fy
end

#--------------- RoeDiffusiveFlux (Lax Wendroff without λ scaling)
struct RoeDiffusiveFlux <: NumericalFluxFunction end

function (lw::RoeDiffusiveFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicPDE{D}) where {D}
    F_L = flux(eq, leftState)
    F_R = flux(eq, rightState)
    
    avg_F = 0.5 * (F_L + F_R)
    diff_U = rightState - leftState

    if abs(diff_U) < 1e-12
        return F_L # or F_R, they are the same
    else
        A_roe_squared_term = (F_L - F_R)^2 / diff_U # This is (-(F_R-F_L))^2 / diff_U = (F_R-F_L)^2 / diff_U
        return avg_F - A_roe_squared_term
    end
end

end
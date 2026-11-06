# Copy of HyperCloud.jl/src/PDEs/euler.jl (Conservative form only), for the probes.
struct EulerEq{D, M, T} <: HyperbolicPDE{D, M, T}
    gamma::T
end
EulerEq(::Val{D}, ::Type{T}, gamma = T(1.4)) where {D, T} = EulerEq{D, D + 2, T}(T(gamma))
@inline function HC.prim2cons(eq::EulerEq{D, M, T}, V::State{M, T}) where {D, M, T}
    rho = V[1]; u = Space{D, T}(ntuple(d -> V[1+d], Val(D))); p = V[M]
    m = rho .* u; E = p / (eq.gamma - T(1)) + T(0.5) * rho * sum(abs2, u)
    return State{M, T}(rho, m..., E)
end
@inline function HC.cons2prim(eq::EulerEq{D, M, T}, U::State{M, T}) where {D, M, T}
    rho = max(U[1], T(1e-7)); m = Space{D, T}(ntuple(d -> U[1+d], Val(D))); E = U[M]
    u = m ./ rho; p = (eq.gamma - T(1)) * (E - T(0.5) * sum(abs2, m) / rho)
    return State{M, T}(rho, u..., max(p, T(1e-7)))
end
@inline function HC.flux(eq::EulerEq{D, M, T}, U::State{M, T}) where {D, M, T}
    V = HC.cons2prim(eq, U); rho = V[1]; u = Space{D, T}(ntuple(d -> V[1+d], Val(D))); p = V[M]; E = U[M]
    return Flux{D, M, T}(ntuple(Val(D)) do d
        ud = u[d]
        State{M, T}(rho * ud, ntuple(i -> rho * u[i] * ud + (i == d ? p : zero(T)), Val(D))..., ud * (E + p))
    end)
end
@inline function HC.is_admissible(eq::EulerEq{D, M, T}, U::State{M, T}) where {D, M, T}
    all(isfinite, U) || return false
    rho = U[1]; rho > zero(T) || return false
    m_sq = sum(abs2, ntuple(i -> U[1+i], Val(D)))
    return (eq.gamma - one(T)) * (U[M] - T(0.5) * m_sq / rho) > zero(T)
end
@inline function HC.max_eigenvalue(eq::EulerEq{D, M, T}, U::State{M, T}, d::Int) where {D, M, T}
    rho = max(U[1], T(1e-9)); m_sq = sum(abs2, ntuple(i -> U[1+i], Val(D)))
    p = max((eq.gamma - one(T)) * (U[M] - T(0.5) * m_sq / rho), T(1e-9))
    return abs(U[1+d] / rho) + sqrt(eq.gamma * p / rho)
end
pressure(eq::EulerEq{D, M, T}, U) where {D, M, T} = (eq.gamma - 1) * (U[M] - 0.5 * sum(abs2, ntuple(i -> U[1+i], Val(D))) / U[1])

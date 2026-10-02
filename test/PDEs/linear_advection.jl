# Notice the addition of the L parameter in the struct and the vel field
struct LinearAdvection{D, M, T, L, R} <: HyperbolicPDE{D, M, T, R}
    vel::Velocity{D, M, T, L}
    max_eigs::SVector{D, T}
    rep::R
end

# Smart constructor
function LinearAdvection(velocities, ::Type{T}=eltype(velocities[1]); rep::R = Conservative()) where {T, R <: EquationRepresentation}
    vel_svec = param2vel(velocities, T)
    
    D = length(vel_svec)
    M = size(vel_svec[1], 1)
    L = M * M # Extract the total length parameter
    
    # Precompute the exact spectral radius
    max_eigs = SVector{D, T}(ntuple(d -> T(maximum(abs.(eigvals(Matrix(vel_svec[d]))))), Val(D)))
    
    return LinearAdvection{D, M, T, L, R}(vel_svec, max_eigs, rep)
end

@inline prim2cons(::LinearAdvection, U::State) = U
@inline cons2prim(::LinearAdvection, W::State) = W

@inline function flux(eq::LinearAdvection{D, M, T}, U::State) where {D, M, T}
    return Flux{D, M, T}(ntuple(Val(D)) do d
        State{M, T}(eq.vel[d] * U)
    end)
end

@inline function max_eigenvalue(eq::LinearAdvection, U::State, d::Int)
    return eq.max_eigs[d]
end

# API implementation returning the strict M x M matrix
@inline velocity(eq::LinearAdvection, U::State, d::Int) = eq.vel[d]

# Fulfill the core API for kinetic relaxation speeds
@inline function kinetic_wave_speed(eq::LinearAdvection{D, NK, T, R}, d::Int, k::Int) where {D, NK, T, R}
    return eq.vel[d][k,k]
end

function build_equation(::Val{:linear}, pde_conf::Dict, context::Dict)
    rep = parse_representation(pde_conf)
    T = context[:Type]::DataType
    
    # We pass T securely, and the smart constructor utilizes param2vel internally
    return LinearAdvection(pde_conf[:velocities], T; rep=rep) 
end

# ---------------------------------------------------------
# Analytic Closures
# ---------------------------------------------------------

function analytic_closure(eq::LinearAdvection{D, M, T}, ic::InitialCondition, geom::GeometricDomain) where {D, M, T}
    # 1. Safety Check: Ensure multi-dimensional system matrices commute
    if D > 1
        for i in 1:D
            for j in (i+1):D
                if !isapprox(eq.vel[i] * eq.vel[j], eq.vel[j] * eq.vel[i]; atol=1e-12)
                    error("Velocity matrices for dimensions $i and $j do not commute. Analytic ray-tracing closure is not mathematically valid for this system.")
                end
            end
        end
    end

    mins = geom.mins
    maxs = geom.maxs
    is_per = geom.is_periodic
    
    # 2. System Eigendecomposition (Safe to sum since they commute)
    vel_sum = sum(eq.vel)
    F = eigen(Matrix(vel_sum)) 
    
    # Force the eigen decomposition back into the requested precision T
    R = SMatrix{M, M, T}(real.(F.vectors))
    L = inv(R)
    
    wave_speeds = SVector{M, SVector{D, T}}(ntuple(Val(M)) do m
        SVector{D, T}(ntuple(Val(D)) do d
            (L * eq.vel[d] * R)[m, m]
        end)
    end)
    
    return function exact_linear_system(st::SVector)
        # Ensure spacetime vector inputs are properly cast to T
        t = T(st[end])
        pos = SVector{D, T}(ntuple(d -> T(st[d]), Val(D))) 
        
        u_final = zeros(MVector{M, T})
        
        # 3. Characteristic Tracing
        for m in 1:M
            vel_m = wave_speeds[m]
            
            pos0 = pos - vel_m * t
            
            pos0 = SVector{D, T}(ntuple(Val(D)) do d
                if is_per[d]
                    mins[d] + mod(pos0[d] - mins[d], maxs[d] - mins[d])
                else
                    pos0[d]
                end
            end)
            
            u0 = ic(pos0)
            
            # Project into characteristic variables
            w_m = zero(T)
            for k in 1:M
                w_m += L[m, k] * u0[k]
            end
            
            # Reconstruct into physical variables
            for k in 1:M
                u_final[k] += R[k, m] * w_m
            end
        end
        
        return State{M, T}(Tuple(u_final))
    end
end

export flux, max_eigenvalue, prim2cons, cons2prim, velocity, update_size!, update_content!, _extract_order

# =========================================================================
# PDE API -> Must be set for every PDE
# =========================================================================

@inline flux(eq::HyperbolicPDE, u::State) = error("flux not implemented for $(typeof(eq))")
@inline max_eigenvalue(eq::HyperbolicPDE, u::State, dim::Int) = error("max_eigenvalue not implemented for $(typeof(eq))")

# These can default to identity if the PDE doesn't use primitive forms
@inline prim2cons(eq::HyperbolicPDE, u::State) = u
@inline cons2prim(eq::HyperbolicPDE, u::State) = u

# Used by the Upwind Flux (for M=1) or Custom Grid Movers
@inline velocity(eq::HyperbolicPDE, u::State, d::Int) = error("velocity not implemented for $(typeof(eq))")

# =========================================================================
# INTERPOLATOR API -> Must be set for every DivergenceInterpolator
# =========================================================================
update_size!(div::DivergenceInterpolator, N_particles::Int) =  error("Size update of the buffers has to be set! Set no-op for stateless interpolators!")
update_content!(div::DivergenceInterpolator, nb_slice, pg, ib) = error("Content update of the buffers has to be set! Set no-op for stateless interpolators! ")
@inline _extract_order(div::DivergenceInterpolator) = error("Order of the method has to be defined for CFL calculation!")



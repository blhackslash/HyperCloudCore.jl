function build_kinetic_system(kin_conf::Dict, context::Dict)
    T = context[:Type]::DataType
    D = context[:D]::Int
    NM = context[:M]::Int
    eq_macro = context[:Equation]
    
    if !haskey(kin_conf, :velocities)
        error("Kinetic simulation requested, but no ':velocities' found in the Kinetic configuration namespace.")
    end
    
    relax_config = kin_conf[:velocities]::Tuple
    relax_eps = T(kin_conf[:epsilon])
    relax_indices = kin_conf[:indices]::Vector{Int}
    
    km = Kin2Macro(relax_indices)
    eq_kin = LinearAdvection(relax_config, T)
    
    coeffs = State{NM, T}(ntuple(m -> one(T) / T(relax_indices[m+1] - relax_indices[m]), Val(NM)))
    interior_factor = T(get(kin_conf, :interior_factor, D)) 
    
    source_term = eq_macro.rep isa Conservative ? 
        RelaxationSourceTerm(km, relax_eps, coeffs, eq_macro, eq_kin, interior_factor) :
        NonLocalRelaxationSourceTerm(km, relax_eps, coeffs, eq_macro, eq_kin, interior_factor)
        
    return eq_kin, source_term
end
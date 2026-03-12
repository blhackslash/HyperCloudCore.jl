function IMEXButcherTableau(A::M, At::M, c::V, ct::V, b::V) where {M <: AbstractArray{Float64, 2}, V <: AbstractArray{Float64, 1}}
    s = size(A, 1) # Number of stages
    @assert (size(A, 2) == s && size(At, 1) == s && size(At, 2) == s &&
                length(c) == s && length(ct) == s && length(b) == s) "All Butcher tableau components must match number of stages"
    # Your old code had ct[1]==0.0, which is a common convention for explicit part starting with U^n.
    # if s > 0 @assert ct[1] == 0.0 "Convention: First explicit time node ct[1] should be 0" end
    
    # Check A is lower triangular (a_ij = 0 for j > i)
    for i in 1:s, j in (i+1):s
        @assert A[i,j] == 0.0 "Implicit matrix A must be lower triangular."
    end
    # Check At is strictly lower triangular (atilde_ij = 0 for j >= i)
    for i in 1:s, j in i:s # Check elements on and above diagonal
        @assert At[i,j] == 0.0 "Explicit matrix At (Atilde) must be strictly lower triangular."
    end
    IMEXButcherTableau{M, V}(A, At, c, ct, b, b)
end

function IMEXButcherTableau(A::M, At::M, c::V, ct::V, b::V, bt::V) where {M <: AbstractArray{Float64, 2}, V <: AbstractArray{Float64, 1}}
    s = size(A, 1) # Number of stages
    @assert (size(A, 2) == s && size(At, 1) == s && size(At, 2) == s &&
                length(c) == s && length(ct) == s && length(b) == s && length(bt) == s) "All Butcher tableau components must match number of stages"    
    # Check A is lower triangular (a_ij = 0 for j > i)
    for i in 1:s, j in (i+1):s
        @assert A[i,j] == 0.0 "Implicit matrix A must be lower triangular."
    end
    # Check At is strictly lower triangular (atilde_ij = 0 for j >= i)
    for i in 1:s, j in i:s # Check elements on and above diagonal
        @assert At[i,j] == 0.0 "Explicit matrix At (Atilde) must be strictly lower triangular."
    end
    IMEXButcherTableau{M, V}(A, At, c, ct, b, bt)
end

function IMEXARS233ButcherTableau(gamma_val::Float64 = (3.0 + sqrt(3.0))/6.0)::IMEXButcherTableau
    # This is the ARS(2,3,3) scheme from Ascher, Ruuth, Spiteri (1997), Table 2.4, k=3 column.
    # It is third order.
    A_impl = [0.0      0.0             0.0;
              0.0      gamma_val       0.0;
              0.0      1.0-2*gamma_val gamma_val]
    # A_impl = [0.0      0.0             0.0;
    #           0.0      0.       0.0;
    #           0.0      0. 0.]

    At_expl = [0.0          0.0              0.0;
               gamma_val    0.0              0.0;
               gamma_val-1.0 2.0*(1-gamma_val) 0.0]

    # In the ARS(k) schemes from Ascher, Ruuth, Spiteri (1997), Table 2.4,
    # the c_i for the implicit part and c_tilde_i for the explicit part are the same.
    c_nodes = [0.0; gamma_val; 1.0-gamma_val] 
    
    # The b_i weights for explicit and implicit parts are also the same.
    b_weights = [0.0; 0.5; 0.5]

    return IMEXButcherTableau(A_impl, At_expl, c_nodes, c_nodes, b_weights)
end

# In a file like IMEXTableaus.jl


function RalstonRK2ButcherTableau()::IMEXButcherTableau

    A_impl = [0. 0.
              0. 0.]
    At_expl = [0. 0.
               (2. /3.) 0.]

    c_nodes = [0, 2. /3.]
    b_weights = [1/4, 3/4]

    return IMEXButcherTableau(A_impl, At_expl, c_nodes, c_nodes, b_weights)
end


function PR_IMEX_SSP3_ButcherTableau()::IMEXButcherTableau
    # Coefficients for Pareschi & Russo (2005), Scheme (4.2)
    # Explicit part (corresponds to SSPRK(3,3) by Shu-Osher)
    ct = [0.0; 1.0; 0.5] # \tilde{c}
    At = [ 0.0   0.0   0.0;
           1.0   0.0   0.0;
           0.25  0.25  0.0 ] # \tilde{A}

    # Implicit part (L-stable SDIRK3)
    # gamma0 is the real root of x^3 - x^2 + x/2 - 1/6 = 0
    gamma0 = 0.24169906235535784649 # Approximate value, can be found with Roots.jl for higher precision
                                    # using Roots; f_g = x -> x^3 - x^2 + x/2 - 1/6; find_zero(f_g, (0.2,0.3))

    c_impl = [gamma0; 
              1.0 - gamma0; 
              0.5] 

    A_impl = zeros(Float64, 3, 3)
    A_impl[1,1] = gamma0
    A_impl[2,1] = 1.0 - 2.0*gamma0
    A_impl[2,2] = gamma0
    A_impl[1,1] = gamma0
    A_impl[2,1] = 1.0 - 2.0*gamma0
    A_impl[2,2] = gamma0
    A_impl[3,1] = ( (1.0-gamma0)/(1.0-2.0*gamma0) - 1.0/(12.0*gamma0) ) * 0.5 # This is (a31_paper + a32_paper_implicit_part_of_FE)
    b1_s_coeff = ( (1.0-gamma0)/(1.0-2.0*gamma0) - 1.0/(12.0*gamma0) ) / 2.0
    b2_s_coeff = (1.0 / (12.0*gamma0*(1.0-2.0*gamma0)) ) / 2.0
    A_impl[3,1] = b1_s_coeff
    A_impl[3,2] = b2_s_coeff
    A_impl[3,3] = gamma0
    
    # Weights (same for explicit and implicit parts in this scheme, matching SSPRK3)
    b_weights = [1.0/6.0; 1.0/6.0; 2.0/3.0]

    return IMEXButcherTableau(A_impl, At, c_impl, ct, b_weights)
end
"""
    ARS222_ButcherTableau(gamma_val::Union{Float64, Nothing}=nothing)::IMEXButcherTableau

Returns the Butcher tableau for the ARS(2,2,2) IMEX scheme from
Ascher, Ruuth, Spiteri (1997), Table 2.2.
This is a 2-stage, 2nd order, L-stable scheme.

The default `gamma_val` is `1.0 - 1.0 / sqrt(2.0)`.
"""
function ARS222_ButcherTableau(gamma_val::Union{Float64, Nothing}=nothing)::IMEXButcherTableau
    # Default gamma for this specific ARS(2,2,2) scheme
    g_coeff = isnothing(gamma_val) ? (1.0 - 1.0 / sqrt(2.0)) : gamma_val
    delta = 1 - 1/(2*g_coeff)
    #g_coeff = 1
    # Explicit Part Coefficients (Atilde, ctilde)
    At_expl = [ 0.0      0.0;
                g_coeff  0.0 ]
    ct_expl = [ 0.0; g_coeff ]

    # Implicit Part Coefficients (A, c)
    A_impl = [ g_coeff            0.0;
               1.0 - g_coeff  g_coeff ]
    c_impl = [ g_coeff; 1.0 ]
    
    # Final Weights (b for both explicit and implicit parts)
    #b_weights = [ 0.5; 0.5 ]
    b = [1-g_coeff; g_coeff] # Implicit b
    bt = [delta; 1-delta]

    return IMEXButcherTableau(A_impl, At_expl, c_impl, ct_expl, b, bt)
end

function SSP2332ButcherTableau()::IMEXButcherTableau
    # SSP2(3, 3, 2) stiffly Accurate scheme
    A = [0.25 0 0; 0 0.25 0; 1/3 1/3 1/3]
    At = [0 0 0; 0.5 0 0; 0.5 0.5 0.0]
    c = [0.25; 0.25; 1.0]
    ct = [0; 0.5; 1]
    b = [1/3; 1/3; 1/3]
    return IMEXButcherTableau(A, At, c, ct, b)
end
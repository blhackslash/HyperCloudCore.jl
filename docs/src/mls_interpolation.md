# Moving Least Squares (MLS) Interpolation

This module implements a universal Moving Least Squares (MLS) interpolator used to construct high-order spatial derivatives for meshfree particle methods. The implementation heavily leverages compile-time metadata and static arrays to ensure that the reconstruction is highly performant and allocation-free.

## 1. Mathematical Background: The Weighted Normal Equations

The objective of the Moving Least Squares method is to find a local polynomial approximation of a field (such as a state difference or flux difference) around a target particle $i$. Let $p(\vec{x})$ be a vector of polynomial basis functions and $\mathbf{c}$ be the vector of unknown coefficients we want to determine. 

For a given particle $i$ interacting with a set of neighbors $j$, we want to minimize the weighted least squares error functional:

$$J(\mathbf{c}) = \sum_{j \in \text{neighbors}} w_j \left( p(\vec{x}_{ij})^T \mathbf{c} - \Delta f_j \right)^2$$

Where:

*   $w_j$ is the distance-based weight of neighbor $j$.
*   $\vec{x}_{ij}$ is the relative distance vector between particle $i$ and neighbor $j$.
*   $\Delta f_j$ is the known data value at neighbor $j$.

To find the minimum, we take the derivative of $J(\mathbf{c})$ with respect to the coefficients $\mathbf{c}$ and set it to zero. This yields the classical weighted Normal Equations:

$$\left( \sum_{j} w_j p(\vec{x}_{ij}) p(\vec{x}_{ij})^T \right) \mathbf{c} = \sum_{j} w_j p(\vec{x}_{ij}) \Delta f_j$$

We can express this compactly as a linear system:

$$N \mathbf{c} = \mathbf{b}$$

Where $N$ is the moment matrix and $\mathbf{b}$ is the right-hand side vector. Once $\mathbf{c}$ is solved, it provides the spatial derivatives of the field directly evaluated at the particle's location.

---

## 2. Implementation Details (`_mls_solve`)

The core of the MLS solver is implemented in the `_mls_solve` function, which assembles and solves the normal equations.

### Assembly and Preconditioning
To ensure the moment matrix $N$ is well-conditioned, the system dynamically rescales the geometry.

*   **Geometric Scaling:** The relative distance vectors are scaled by an inverse length factor `invL = one(T) / scale` before evaluating the basis polynomials.
*   **Taylor Prefactors:** The `build_basis` function calculates factorial denominators at compile-time to construct a proper Taylor basis, which keeps the magnitude of higher-order terms constrained.
*   **Matrix Assembly:** The code initializes the moment matrix `N_s` and the right-hand side `b_s` as zero-filled `SMatrix` types. It then loops over the neighbor slice, adding the weighted outer products `w * (p_s * p_s')` to `N_s` and `w * (p_s * dfVec[i]')` to `b_s`.

### Allocation-Free Cholesky Decomposition
Solving the normal equations using a QR decomposition on the original design matrix is generally more numerically stable, but it requires dynamic memory allocations or significantly larger matrix operations. To maintain strict allocation-free performance, this package solves the normal equations directly using a Cholesky decomposition.

*   **Static Arrays:** The linear system uses `SMatrix` arrays whose sizes (`B_LEN`) are resolved dynamically at compile time via `basis_length`. 
*   **Symmetric Cholesky:** The system invokes `cholesky(Symmetric(N_s_reg), check=false)` directly on the statically sized moment matrix. Because the dimensions are known to the compiler, the Cholesky factorization is fully unrolled and allocation-free.
*   **Regularization:** Because solving the normal equations squares the condition number of the problem, the formulation can become poorly conditioned. To mitigate this, a small regularization term is added to the diagonal: `eps_reg = T(1e-10) * tr(N_s) / B_LEN`. The stabilized matrix `N_s_reg = N_s + eps_reg * I` is then factorized.
*   **Fallback:** The code checks `LinearAlgebra.issuccess(C)`. If the decomposition fails (e.g., due to a highly degenerate particle topology), it gracefully falls back to returning a zero vector.

### Final State Reconstruction
After the coefficient matrix `c_s` is found via backslash division (`C \ b_s`), the geometric scaling is reverted. The code evaluates `build_scale_factors` and multiplies the coefficients by their respective physical scale factors to yield the true derivatives.
# Full Potential Solver - Architecture Documentation

## Overview

This is a production-quality **transonic full potential solver** implemented in Chapel (HPC language) for compressible, inviscid flow around lifting bodies. The solver is designed for transonic airfoil analysis, optimization studies, and design applications requiring accurate pressure distributions and aerodynamic coefficients.

## Mathematical Foundation

### Governing Equation

The solver implements the **full potential equation** with an **isentropic assumption**:

```
∇·(ρ∇φ) = 0
```

where:
- **φ** = velocity potential
- **ρ** = density (function of velocity magnitude)
- **u = ∇φ** = velocity field

### Isentropic Density-Velocity Relation

From Bernoulli equation and isentropic relations:

```
ρ = [1 + ((γ-1)/2) × M²∞ × (1 - |u|²)]^(1/(γ-1))
```

This coupling creates a **nonlinear PDE** in φ.

### Kutta Condition for Lifting Flows

For lifting bodies, the solver implements circulation (Γ) to satisfy the Kutta condition:

- Identifies trailing edge (TE) location
- Defines wake streamline downstream of TE
- Enforces: **φ_upper_TE - φ_lower_TE = Γ**
- Circulation becomes an additional degree of freedom solved simultaneously

The complete system becomes:
```
R(φ, Γ) = 0
```
where R includes:
- Continuity residuals for all cells
- Kutta condition: Γ - (φ_upper - φ_lower) = 0

## Project Structure

```
fullPotentialSolver/
├── src/                    # Chapel source code (computational kernel)
│   ├── main.chpl          # Entry point, orchestration
│   ├── mesh.chpl          # CGNS reader, connectivity builder
│   ├── spatialDiscretization.chpl   # Geometry, gradients, fluxes
│   ├── temporalDiscretization.chpl  # Newton solver, Jacobian
│   ├── leastSquaresGradient.chpl    # QR-based gradient reconstruction
│   ├── linearAlgebra.chpl           # PETSc GMRES interface
│   ├── writeCGNS.chpl              # Solution output
│   └── input.chpl                   # Configuration management
├── pre/                   # Python mesh generation scripts
│   ├── cylinder.py        # O-grid for cylinder
│   └── naca_ibm.py       # NACA airfoil mesh
├── post/                  # Python post-processing
│   └── read_cgns.py      # Cp plots, convergence visualization
├── output/               # CGNS solution files
├── doc/                  # LaTeX documentation
└── bin/                  # Compiled executables
```

## Core Modules

### 1. mesh.chpl - Mesh Management

**Responsibilities:**
- Reads CGNS mesh files (HDF5 format)
- Builds connectivity data structures:
  - `edge2node`: edge-to-vertex mapping
  - `elem2edge`: element-to-edge mapping
  - `edge2elem`: edge-to-element mapping (includes ghost cells)
  - `esup1/esup2`: elements around nodes
  - `psup1/psup2`: nodes around nodes
- Creates ghost cells for boundary conditions (mirroring)
- Identifies boundary edges (wall, farfield)

**Key Features:**
- Supports quad and triangle elements
- Uses 1-based indexing (Chapel convention)
- Ghost cells extend domain beyond physical boundaries

### 2. spatialDiscretization.chpl - Spatial Discretization

**Core Responsibilities:**

1. **Geometric metrics computation:**
   - Element centroids and volumes
   - Face centroids, areas, and normals
   - Precomputed flux coefficients for efficiency

2. **Gradient reconstruction:**
   - QR-based least-squares gradients (Blazek formulation)
   - Precomputed weight matrices for fast runtime evaluation
   - Wake-aware gradients with circulation correction

3. **Flux computation:**
   - Deferred correction approach for face values
   - Isentropic density from Bernoulli equation
   - Continuity flux: F = ρ(V·n)A

4. **Residual computation:**
   - Accumulates fluxes per element
   - Computes Kutta condition residual

5. **Boundary conditions:**
   - **Wall:** Zero normal velocity (φ_ghost = φ_interior)
   - **Farfield:** Freestream conditions

6. **Kutta condition implementation:**
   - Identifies cells above/below wake (kuttaCell_ array)
   - Applies circulation jump across wake
   - Extrapolates TE values from face gradients

**Key Data Structures:**
```chapel
phi_[elem]          // Potential at each cell
uu_[elem], vv_[elem] // Velocity components
rho_[elem]          // Density
circulation_        // Global circulation value
kuttaCell_[elem]    // Wake identifier: 1 (above), -1 (below), 9 (elsewhere)
```

### 3. temporalDiscretization.chpl - Nonlinear Solver

**Solution Strategy:**

Uses **Newton's method** to solve the nonlinear system R(φ, Γ) = 0

**Newton Iteration:**
```
J · Δq = -ω·R
q^(n+1) = q^n + Δq
```

where:
- **q = [φ₁, φ₂, ..., φₙ, Γ]ᵀ** (n cells + 1 circulation DOF)
- **J** is the Jacobian matrix (∂R/∂q)
- **ω** is relaxation factor

**Jacobian Computation:**

The Jacobian includes:
1. **∂(Res_i)/∂(φ_j)**: Flux derivatives w.r.t. neighboring potentials
2. **∂(Res_i)/∂Γ**: Circulation effect on wake-adjacent cells
3. **∂(Kutta)/∂(φ_TE)**: Kutta condition dependencies
4. **∂(Kutta)/∂Γ = 1**: Direct circulation dependency

**Key Innovation:** Precomputes **gradient sensitivity to circulation** (∂∇φ/∂Γ) which captures how circulation affects gradients through wake-crossing faces. This is crucial for Newton convergence.

**Linear Solver:**
- Uses PETSc GMRES with ILU/Jacobi/LU preconditioning
- Configurable restart, tolerances
- Sparse matrix storage (CSR format)

### 4. leastSquaresGradient.chpl - Gradient Reconstruction

**Two Implementations:**

1. **LeastSquaresGradient** - Normal equations approach
   - Solves: A·grad = b where A = Σ(w·dx⊗dx)
   - Precomputes and stores A⁻¹ per cell

2. **LeastSquaresGradientQR** (used in solver) - QR factorization
   - More stable on stretched meshes
   - Weights: θ_j = 1/d_ij (inverse distance)
   - Fully precomputed weights per face for both cell perspectives
   - Runtime: grad_i = Σ_j [wx, wy]_ij × (φ_j - φ_i)

**Wake Handling:**

Applies circulation correction when computing differences across wake:
```chapel
if (elemAbove && neighborBelow) then phiJ += circulation
if (elemBelow && neighborAbove) then phiJ -= circulation
```

### 5. linearAlgebra.chpl - Linear Solvers

Provides:
- `GMRES()` - Interface to PETSc GMRES
- `RMSE()` - Root mean square error computation (volume-weighted)
- Iterative solvers (Jacobi, Gauss-Seidel) for debugging

### 6. input.chpl - Configuration Management

Reads parameters from `input.txt`:
- Mesh file and element type
- Flow conditions (Mach, alpha, gamma)
- Solver parameters (CFL, iterations, tolerances)
- GMRES settings
- Circulation relaxation factor

### 7. writeCGNS.chpl - Solution Output

Writes CGNS files containing:
- Mesh (coordinates, connectivity)
- Cell-centered solution fields (φ, u, v, ρ, Cp, Mach, residuals)
- Wall boundary quantities
- Convergence history (iterations, residuals, CL, CD, CM, circulation)

## Numerical Methods

### Spatial Discretization - Finite Volume

**Face Flux with Deferred Correction:**

The face velocity uses a sophisticated deferred correction scheme:

```
V_face = V_avg - δ × corrCoeff
```

where:
- `V_avg = 0.5×(∇φ₁ + ∇φ₂)` (averaged gradients)
- `δ = V_avg·t_IJ - (φ₂ - φ₁)/L_IJ` (correction term)
- `corrCoeff = n/(n·t_IJ)` (geometric correction coefficient)
- `t_IJ` = unit vector from cell 1 to cell 2

**Advantages:**
- Ensures consistency with direct potential difference
- Corrects for non-orthogonal meshes
- Precomputed geometric coefficients for efficiency

**Continuity Flux:**
```
F = ρ_face × (u_face×n_x + v_face×n_y) × Area
```

where ρ_face is computed from isentropic relation at face velocity.

**Gradient Reconstruction:**
- QR-based least squares (Blazek method)
- Weighted by inverse distance for accuracy on stretched grids
- Fully precomputed weight matrices
- O(1) runtime per cell

### Temporal Discretization - Newton's Method

**Overall Algorithm:**

```
1. Initialize:
   - Read mesh and build connectivity
   - Compute geometric metrics
   - Identify Kutta cells and TE
   - Initialize φ = U∞·x + V∞·y
   - Precompute Jacobian structure

2. Newton iteration loop (until convergence):
   a. Update ghost cells (BC application)
   b. Compute gradients: u, v = ∇φ (with circulation correction)
   c. Compute density: ρ = ρ(|u|) from isentropic relation
   d. Update ghost velocities (BC for fluxes)
   e. Compute face velocities and densities
   f. Compute fluxes: F = ρ(V·n)A
   g. Accumulate residuals: R = Σ(flux)
   h. Compute Kutta residual: R_Γ = Γ - (φ_upper - φ_lower)

   i. Compute Jacobian: J = ∂R/∂[φ, Γ]
      - Include gradient sensitivity ∂∇φ/∂Γ for wake cells

   j. Solve linear system: J·Δq = -ω·R (using GMRES)

   k. Update: φ ← φ + Δφ, Γ ← Γ + ΔΓ

   l. Compute aerodynamic coefficients (CL, CD, CM)

   m. Check convergence: ||R|| < tolerance

3. Output final solution to CGNS
```

**Convergence Criteria:**
- Residual norm: ||R||/||R₀|| < 1e-8
- Maximum iterations: 30,000
- Monitors separate residuals for wall, fluid, and wake regions

## CGNS File Usage

### Input (Pre-processing)

Python scripts generate structured CGNS meshes:
- **Gmsh** creates geometry and structured grids (O-grid for cylinder)
- Geometric progression for boundary layer refinement
- Exports CGNS format
- **h5py** post-processes to merge elements and rename nodes

**CGNS Structure Expected:**
```
/Base/
  /dom-1/
    /GridCoordinates/
      CoordinateX, CoordinateY, CoordinateZ
    /QuadElements/ (or TriElements)
      ElementConnectivity
    /wall/
      ElementConnectivity
    /farfield/
      ElementConnectivity
```

### Output (Post-processing)

Solver writes CGNS files:

```
/Base/
  /dom-1/
    /FLOW_SOLUTION_CC/ (cell-centered)
      phi, u, v, rho, cp, mach, res, xElem, yElem, kuttaCell
  /GlobalConvergenceHistory/
    IterationCounters, Time, Residual, Cl, Cd, Cm, Circulation
  /WallBase/
    /wall/WALL_FLOW_SOLUTION_CC/
      xWall, yWall, nxWall, nyWall, uWall, vWall, cpWall
```

**Python Post-Processing (read_cgns.py):**
- Reads CGNS files using h5py
- Extracts wall pressure coefficients
- Plots Cp distributions
- Visualizes convergence history
- Compares with reference solutions
- Computes wall normal velocity to verify boundary conditions

## Unique Technical Features

### 1. Full Coupling of Circulation

Unlike methods that iterate between potential solve and circulation update, this solver treats Γ as part of the state vector, solving for [φ, Γ] simultaneously in a single Newton system.

### 2. Jacobian Includes ∂∇φ/∂Γ

The Jacobian computation captures how circulation affects gradients through wake-crossing faces. This is essential for Newton convergence and is a key innovation in the implementation.

### 3. Efficient Precomputation

- Geometric coefficients computed once during initialization
- Gradient weights (QR factorization) precomputed
- Sparse Jacobian pattern precomputed
- Runtime operations are pure arithmetic (no matrix inversions)

### 4. Wake-Aware Gradient Reconstruction

Special handling for wake-crossing faces ensures:
- Proper circulation jump in potential differences
- Accurate gradients near trailing edge
- Smooth transition across wake streamline

### 5. Trailing Edge Extrapolation

Uses face gradients to extrapolate TE values for Kutta BC, providing better accuracy than cell-centered values.

## Implementation Highlights

### Efficiency
- Precomputed geometric coefficients (spatialDiscretization.chpl:144-339)
- Precomputed gradient weights
- Sparse Jacobian storage
- Chapel's parallel `forall` loops for multi-core execution

### Robustness
- QR factorization (more stable than normal equations)
- Relaxation factor on Newton updates
- Multiple preconditioner options
- Convergence monitoring with separate residual norms

### Generality
- Supports quad and triangle meshes
- Unstructured mesh capability
- Multiple boundary condition types
- Configurable solver parameters

### Physical Accuracy
- Kutta condition for circulation
- Isentropic compressibility effects
- Deferred correction for non-orthogonal meshes
- Proper TE extrapolation

## Technology Stack

- **Chapel** - Primary computational language (HPC, parallel computing)
- **Python** - Pre/post-processing
  - h5py - CGNS/HDF5 file I/O
  - matplotlib - Visualization
  - Gmsh API - Mesh generation
- **PETSc** - Linear algebra backend (GMRES solver)
- **CGNS/HDF5** - Standard CFD data format

## Typical Workflow

```
1. Mesh Generation (Python):
   pre/cylinder.py or pre/naca_ibm.py → mesh.cgns

2. Configure Solver:
   Edit input.txt (Mach, alpha, tolerance, etc.)

3. Compile and Run:
   chpl src/main.chpl -o bin/solver
   ./bin/solver

4. Post-process:
   python post/read_cgns.py
   → Cp plots, convergence history, validation
```

## Applications

This solver is suitable for:
- **Transonic airfoil analysis** - Accurate Cp distributions, shock capturing
- **Optimization studies** - Fast convergence for design iterations
- **Design applications** - Aerodynamic coefficient prediction
- **Educational purposes** - Full-featured CFD example in Chapel
- **Research** - Platform for algorithm development

## References

Key algorithms implemented from:
- Blazek, J. "Computational Fluid Dynamics: Principles and Applications" (Least-squares gradients)
- Anderson, J.D. "Fundamentals of Aerodynamics" (Full potential theory)
- Saad & Schultz - GMRES algorithm
- Barth & Jespersen - Gradient reconstruction on unstructured grids

---

**Last Updated:** 2025-12-26
**Version:** Current implementation

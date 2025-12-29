/**
 * Native Chapel GMRES Solver with ILU(0) Preconditioner
 * 
 * Implements the Generalized Minimal Residual method for solving Ax = b
 * with flexible restart and ILU(0) preconditioning for sparse matrices.
 * 
 * Key features:
 * - Full Chapel parallelism via forall loops
 * - Sparse matrix storage in CSR format
 * - ILU(0) preconditioner (computed in-place)
 * - Flexible GMRES with configurable restart
 * - Arnoldi orthogonalization with Modified Gram-Schmidt
 */
module gmres {

use Math;
use Time;

// ============================================================================
// Sparse Matrix in CSR Format
// ============================================================================

record SparseMatrixCSR {
    var n: int;                           // Matrix dimension (n x n)
    var nnz: int;                         // Number of non-zeros
    
    var rowPtr_dom: domain(1);            // Domain for row pointers
    var colIdx_dom: domain(1);            // Domain for column indices and values
    
    var rowPtr: [rowPtr_dom] int;         // Row pointers (size n+1)
    var colIdx: [colIdx_dom] int;         // Column indices
    var values: [colIdx_dom] real(64);    // Non-zero values
    
    // Diagonal element indices for fast access (used by ILU and Jacobi)
    var diagIdx: [rowPtr_dom] int;
    
    proc init(n_: int, nnz_: int) {
        this.n = n_;
        this.nnz = nnz_;
        this.rowPtr_dom = {0..n_};
        this.colIdx_dom = {0..#nnz_};
    }
    
    proc init() {
        this.n = 0;
        this.nnz = 0;
        this.rowPtr_dom = {0..0};
        this.colIdx_dom = {0..#1};
    }
    
    // Build diagonal index lookup after matrix is assembled
    proc ref buildDiagonalIndex() {
        forall i in 0..#n {
            for k in rowPtr[i]..rowPtr[i+1]-1 {
                if colIdx[k] == i {
                    diagIdx[i] = k;
                    break;
                }
            }
        }
    }
    
    // Matrix-vector product: y = A * x
    proc matvec(ref y: [] real(64), const ref x: [] real(64)) {
        forall i in 0..#n {
            var sum = 0.0;
            for k in rowPtr[i]..rowPtr[i+1]-1 {
                sum += values[k] * x[colIdx[k]];
            }
            y[i] = sum;
        }
    }
}

// ============================================================================
// ILU(0) Preconditioner
// ============================================================================

record ILU0Preconditioner {
    var n: int;
    var nnz: int;
    
    var rowPtr_dom: domain(1);
    var colIdx_dom: domain(1);
    
    var rowPtr: [rowPtr_dom] int;
    var colIdx: [colIdx_dom] int;
    var LU: [colIdx_dom] real(64);      // Combined L and U factors
    var diagIdx: [rowPtr_dom] int;       // Index of diagonal in each row
    
    var initialized: bool = false;
    
    proc init() {
        this.n = 0;
        this.nnz = 0;
        this.rowPtr_dom = {0..0};
        this.colIdx_dom = {0..#1};
    }
    
    // Initialize from sparse matrix structure
    proc ref setup(const ref A: SparseMatrixCSR) {
        this.n = A.n;
        this.nnz = A.nnz;
        this.rowPtr_dom = A.rowPtr_dom;
        this.colIdx_dom = A.colIdx_dom;
        
        // Copy structure
        this.rowPtr = A.rowPtr;
        this.colIdx = A.colIdx;
        this.diagIdx = A.diagIdx;
        
        this.initialized = true;
    }
    
    // Compute ILU(0) factorization: A ≈ L * U
    // L has unit diagonal, U has the actual diagonal
    // Both are stored in LU array using the same sparsity as A
    proc ref factorize(const ref A: SparseMatrixCSR) {
        // Copy values
        this.LU = A.values;
        
        // Build column-to-position mapping for each row to accelerate lookups
        // This allows O(1) lookup instead of O(nnz_per_row) linear search
        var colToPos: [0..#n] int = -1;  // Temporary array for current row
        
        // ILU(0) factorization (row-wise, sequential for dependencies)
        for i in 1..n-1 {
            const rowStart = rowPtr[i];
            const rowEnd = rowPtr[i+1];
            const diagPos = diagIdx[i];
            
            // Build column map for row i
            for k in rowStart..rowEnd-1 {
                colToPos[colIdx[k]] = k;
            }
            
            // For each non-zero in lower triangle of row i
            for k in rowStart..diagPos-1 {
                const j = colIdx[k];  // Column index (j < i)
                
                // Get diagonal of row j
                const diagJ = diagIdx[j];
                const diagVal = LU[diagJ];
                if abs(diagVal) < 1e-14 then continue;
                
                // L_ij = A_ij / U_jj
                LU[k] /= diagVal;
                const L_ij = LU[k];
                
                // Update remaining elements in row i: A_ik -= L_ij * U_jk
                // Only update positions that exist in sparsity pattern
                for p in diagJ+1..rowPtr[j+1]-1 {
                    const col_p = colIdx[p];
                    const pos = colToPos[col_p];
                    if pos >= 0 {
                        // Position exists in row i
                        LU[pos] -= L_ij * LU[p];
                    }
                    // Drop fill-in that doesn't match sparsity pattern (ILU(0))
                }
            }
            
            // Clear column map for row i
            for k in rowStart..rowEnd-1 {
                colToPos[colIdx[k]] = -1;
            }
        }
    }
    
    // Solve (LU) * x = b  =>  L * y = b, then U * x = y
    // Forward and backward substitution
    proc solve(ref x: [] real(64), const ref b: [] real(64)) {
        // Forward substitution: L * y = b (y stored in x)
        for i in 0..#n {
            var sum = b[i];
            const rowStart = rowPtr[i];
            const diagPos = diagIdx[i];
            
            for k in rowStart..diagPos-1 {
                sum -= LU[k] * x[colIdx[k]];
            }
            x[i] = sum;  // L has unit diagonal
        }
        
        // Backward substitution: U * x = y
        for i in (0..#n by -1) {
            const diagPos = diagIdx[i];
            const rowEnd = rowPtr[i+1];
            
            var sum = x[i];
            for k in diagPos+1..rowEnd-1 {
                sum -= LU[k] * x[colIdx[k]];
            }
            x[i] = sum / LU[diagPos];
        }
    }
}

// ============================================================================
// Jacobi Preconditioner (simpler, parallel)
// ============================================================================

record JacobiPreconditioner {
    var n: int;
    var invDiag_dom: domain(1);
    var invDiag: [invDiag_dom] real(64);
    
    proc init() {
        this.n = 0;
        this.invDiag_dom = {0..#1};
    }
    
    proc ref setup(const ref A: SparseMatrixCSR) {
        this.n = A.n;
        this.invDiag_dom = {0..#A.n};
        
        forall i in 0..#n {
            const diagVal = A.values[A.diagIdx[i]];
            invDiag[i] = if abs(diagVal) > 1e-14 then 1.0 / diagVal else 1.0;
        }
    }
    
    // Solve: M * x = b  where M = diag(A)
    proc solve(ref x: [] real(64), const ref b: [] real(64)) {
        forall i in 0..#n {
            x[i] = invDiag[i] * b[i];
        }
    }
}

// ============================================================================
// GMRES Solver
// ============================================================================

enum PreconditionerType { None, Jacobi, ILU0 }
enum PreconSide { Left, Right }
enum GMRESConvergedReason {
    Converged_RTOL = 2,
    Converged_ATOL = 3,
    Diverged_MaxIt = -3,
    Diverged_Breakdown = -5,
    NotConverged = 0
}

record GMRESSolver {
    var n: int;                           // System size
    var restart: int;                     // Restart parameter (m)
    var maxIter: int;                     // Maximum iterations
    var rtol: real(64);                   // Relative tolerance
    var atol: real(64);                   // Absolute tolerance
    
    var preconType: PreconditionerType;
    var preconSide: PreconSide;
    var jacobiPrecon: JacobiPreconditioner;
    var iluPrecon: ILU0Preconditioner;
    var preconInitialized: bool = false;  // Track if preconditioner is set up
    
    // Work arrays
    var work_dom: domain(1);
    var r: [work_dom] real(64);           // Residual
    var w: [work_dom] real(64);           // Work vector
    var z: [work_dom] real(64);           // Preconditioned vector
    
    // Krylov basis V (n x (m+1))
    var V_dom: domain(2);
    var V: [V_dom] real(64);
    
    // Hessenberg matrix H ((m+1) x m)
    var H_dom: domain(2);
    var H: [H_dom] real(64);
    
    // Givens rotation parameters
    var cs_dom: domain(1);
    var cs: [cs_dom] real(64);
    var sn: [cs_dom] real(64);
    
    // Right-hand side for least squares
    var g: [cs_dom] real(64);
    
    // Solution in Krylov basis
    var y: [cs_dom] real(64);
    
    // Default initializer (needed for record fields with default init)
    proc init() {
        this.n = 0;
        this.restart = 30;
        this.maxIter = 1000;
        this.rtol = 1e-6;
        this.atol = 1e-12;
        this.preconType = PreconditionerType.None;
        this.preconSide = PreconSide.Right;
        this.work_dom = {0..#1};
        this.V_dom = {0..#1, 0..1};
        this.H_dom = {0..1, 0..#1};
        this.cs_dom = {0..1};
    }
    
    proc init(n_: int, restart_: int = 30, maxIter_: int = 1000,
              rtol_: real(64) = 1e-6, atol_: real(64) = 1e-12,
              preconType_: PreconditionerType = PreconditionerType.None,
              preconSide_: PreconSide = PreconSide.Right) {
        this.n = n_;
        this.restart = restart_;
        this.maxIter = maxIter_;
        this.rtol = rtol_;
        this.atol = atol_;
        this.preconType = preconType_;
        this.preconSide = preconSide_;
        
        this.work_dom = {0..#n_};
        this.V_dom = {0..#n_, 0..restart_};
        this.H_dom = {0..restart_, 0..#restart_};
        this.cs_dom = {0..restart_};
    }
    
    proc ref setupPreconditioner(const ref A: SparseMatrixCSR) {
        select preconType {
            when PreconditionerType.Jacobi {
                jacobiPrecon.setup(A);
            }
            when PreconditionerType.ILU0 {
                iluPrecon.setup(A);
                iluPrecon.factorize(A);
            }
            otherwise { }
        }
        preconInitialized = true;
    }
    
    // Update preconditioner values only (skip setup if already initialized)
    proc ref updatePreconditioner(const ref A: SparseMatrixCSR) {
        if !preconInitialized {
            setupPreconditioner(A);
            return;
        }
        
        select preconType {
            when PreconditionerType.ILU0 {
                // Only refactorize, skip setup
                iluPrecon.factorize(A);
            }
            when PreconditionerType.Jacobi {
                // Jacobi is cheap, always update
                jacobiPrecon.setup(A);
            }
            otherwise { }
        }
    }
    
    // Set tolerance dynamically (for inexact Newton)
    proc ref setTolerance(rtol_new: real(64), atol_new: real(64)) {
        this.rtol = rtol_new;
        this.atol = atol_new;
    }
    
    // Apply preconditioner: z = M^{-1} * r
    proc applyPreconditioner(ref z_: [] real(64), const ref r_: [] real(64)) {
        select preconType {
            when PreconditionerType.Jacobi {
                jacobiPrecon.solve(z_, r_);
            }
            when PreconditionerType.ILU0 {
                iluPrecon.solve(z_, r_);
            }
            otherwise {
                forall i in 0..#n do z_[i] = r_[i];
            }
        }
    }
    
    // Compute 2-norm
    proc norm2(const ref v: [] real(64)): real(64) {
        var sum = 0.0;
        forall val in v with (+ reduce sum) {
            sum += val * val;
        }
        return sqrt(sum);
    }
    
    // Dot product
    proc dot(const ref a: [] real(64), const ref b: [] real(64)): real(64) {
        var sum = 0.0;
        forall (ai, bi) in zip(a, b) with (+ reduce sum) {
            sum += ai * bi;
        }
        return sum;
    }
    
    // Apply Givens rotation
    proc applyGivensRotation(ref h1: real(64), ref h2: real(64), 
                             cs_: real(64), sn_: real(64)) {
        const temp = cs_ * h1 + sn_ * h2;
        h2 = -sn_ * h1 + cs_ * h2;
        h1 = temp;
    }
    
    // Generate Givens rotation to eliminate h2
    proc generateGivensRotation(h1: real(64), h2: real(64),
                                 ref cs_: real(64), ref sn_: real(64)) {
        if h2 == 0.0 {
            cs_ = 1.0;
            sn_ = 0.0;
        } else if abs(h2) > abs(h1) {
            const t = h1 / h2;
            sn_ = 1.0 / sqrt(1.0 + t * t);
            cs_ = t * sn_;
        } else {
            const t = h2 / h1;
            cs_ = 1.0 / sqrt(1.0 + t * t);
            sn_ = t * cs_;
        }
    }
    
    // Main GMRES solve: solve A * x = b
    // Supports both LEFT and RIGHT preconditioning:
    // - LEFT:  solve M^{-1}*A*x = M^{-1}*b (monitors preconditioned residual)
    // - RIGHT: solve A*M^{-1}*u = b, then x = M^{-1}*u (monitors TRUE residual)
    // updatePrecon: if false, reuse existing preconditioner (for lagging)
    proc ref solve(const ref A: SparseMatrixCSR, 
                   ref x: [] real(64), 
                   const ref b: [] real(64),
                   updatePrecon: bool = true): (int, GMRESConvergedReason) {
        
        if preconSide == PreconSide.Right {
            return solveRightPrecon(A, x, b, updatePrecon);
        } else {
            return solveLeftPrecon(A, x, b, updatePrecon);
        }
    }
    
    // LEFT preconditioning: solve M^{-1}*A*x = M^{-1}*b
    // Monitors preconditioned residual ||M^{-1}(b - Ax)||
    proc ref solveLeftPrecon(const ref A: SparseMatrixCSR, 
                              ref x: [] real(64), 
                              const ref b: [] real(64),
                              updatePrecon: bool = true): (int, GMRESConvergedReason) {
        
        // Update or setup preconditioner
        if updatePrecon {
            if preconInitialized {
                updatePreconditioner(A);
            } else {
                setupPreconditioner(A);
            }
        } else if !preconInitialized {
            // First time, must setup
            setupPreconditioner(A);
        }
        
        var totalIter = 0;
        var reason = GMRESConvergedReason.NotConverged;
        
        // Compute initial residual: r = b - A * x
        A.matvec(r, x);
        forall i in 0..#n do r[i] = b[i] - r[i];
        
        // Apply preconditioner to get z = M^{-1} * r
        applyPreconditioner(z, r);
        
        const bnorm = norm2(b);
        var rnorm = norm2(z);  // Preconditioned residual norm
        const rnorm0 = rnorm;
        
        // Check if already converged
        if rnorm < atol || (bnorm > 0.0 && rnorm / bnorm < rtol) {
            return (0, GMRESConvergedReason.Converged_ATOL);
        }
        
        // Outer restart loop
        while totalIter < maxIter {
            if rnorm < 1e-14 {
                reason = GMRESConvergedReason.Diverged_Breakdown;
                break;
            }
            
            // LEFT PRECONDITIONING: V[:,0] = z / ||z|| (preconditioned residual)
            const invRnorm = 1.0 / rnorm;
            forall i in 0..#n do V[i, 0] = z[i] * invRnorm;
            
            // Initialize g = [||z||, 0, 0, ...]
            g[0] = rnorm;
            for j in 1..restart do g[j] = 0.0;
            
            // Arnoldi iteration
            var converged = false;
            var j = 0;
            
            while j < restart && totalIter < maxIter {
                totalIter += 1;
                
                // LEFT PRECONDITIONING: w = M^{-1} * A * V[:,j]
                // Step 1: z = A * V[:,j]
                forall i in 0..#n do z[i] = V[i, j];
                A.matvec(w, z);
                
                // Step 2: z = M^{-1} * w
                applyPreconditioner(z, w);
                
                // Copy z to w for orthogonalization
                forall i in 0..#n do w[i] = z[i];
                
                // Modified Gram-Schmidt orthogonalization
                for i in 0..j {
                    var hij = 0.0;
                    forall k in 0..#n with (+ reduce hij) {
                        hij += w[k] * V[k, i];
                    }
                    H[i, j] = hij;
                    
                    forall k in 0..#n {
                        w[k] -= hij * V[k, i];
                    }
                }
                
                // H[j+1, j] = ||w||
                const wnorm = norm2(w);
                H[j+1, j] = wnorm;
                
                // Check for breakdown
                if wnorm < 1e-14 {
                    reason = GMRESConvergedReason.Diverged_Breakdown;
                    converged = true;
                    break;
                }
                
                // V[:,j+1] = w / ||w||
                const invWnorm = 1.0 / wnorm;
                forall i in 0..#n do V[i, j+1] = w[i] * invWnorm;
                
                // Apply previous Givens rotations to H[:,j]
                for i in 0..j-1 {
                    applyGivensRotation(H[i, j], H[i+1, j], cs[i], sn[i]);
                }
                
                // Generate new Givens rotation
                generateGivensRotation(H[j, j], H[j+1, j], cs[j], sn[j]);
                
                // Apply new rotation
                applyGivensRotation(H[j, j], H[j+1, j], cs[j], sn[j]);
                applyGivensRotation(g[j], g[j+1], cs[j], sn[j]);
                
                // Check convergence: |g[j+1]| is the preconditioned residual norm
                rnorm = abs(g[j+1]);
                
                if rnorm < atol {
                    reason = GMRESConvergedReason.Converged_ATOL;
                    converged = true;
                    j += 1;
                    break;
                }
                if bnorm > 0.0 && rnorm / bnorm < rtol {
                    reason = GMRESConvergedReason.Converged_RTOL;
                    converged = true;
                    j += 1;
                    break;
                }
                if rnorm0 > 0.0 && rnorm / rnorm0 < rtol {
                    reason = GMRESConvergedReason.Converged_RTOL;
                    converged = true;
                    j += 1;
                    break;
                }
                
                j += 1;
            }
            
            // Solve upper triangular system H * y = g
            const m = j;  // Number of Arnoldi steps completed
            for i in (0..m-1 by -1) {
                y[i] = g[i];
                for k in i+1..m-1 {
                    y[i] -= H[i, k] * y[k];
                }
                y[i] /= H[i, i];
            }
            
            // LEFT PRECONDITIONING: x = x + V * y (no preconditioner on solution)
            forall i in 0..#n {
                var sum = 0.0;
                for k in 0..m-1 {
                    sum += V[i, k] * y[k];
                }
                x[i] += sum;
            }
            
            if converged then break;
            
            // Compute new residual for restart: r = b - A * x
            A.matvec(r, x);
            forall i in 0..#n do r[i] = b[i] - r[i];
            
            // Apply preconditioner for next iteration
            applyPreconditioner(z, r);
            rnorm = norm2(z);
            
            // Check convergence after restart
            if rnorm < atol {
                reason = GMRESConvergedReason.Converged_ATOL;
                break;
            }
            if bnorm > 0.0 && rnorm / bnorm < rtol {
                reason = GMRESConvergedReason.Converged_RTOL;
                break;
            }
        }
        
        if totalIter >= maxIter && reason == GMRESConvergedReason.NotConverged {
            reason = GMRESConvergedReason.Diverged_MaxIt;
        }
        
        return (totalIter, reason);
    }
    
    // RIGHT preconditioning: solve A*M^{-1}*u = b, then x = M^{-1}*u
    // Monitors TRUE residual ||b - Ax|| (matches PETSc default)
    proc ref solveRightPrecon(const ref A: SparseMatrixCSR, 
                               ref x: [] real(64), 
                               const ref b: [] real(64),
                               updatePrecon: bool = true): (int, GMRESConvergedReason) {
        
        // Update or setup preconditioner
        if updatePrecon {
            if preconInitialized {
                updatePreconditioner(A);
            } else {
                setupPreconditioner(A);
            }
        } else if !preconInitialized {
            // First time, must setup
            setupPreconditioner(A);
        }
        
        var totalIter = 0;
        var reason = GMRESConvergedReason.NotConverged;
        
        // Compute initial residual: r = b - A * x
        A.matvec(r, x);
        forall i in 0..#n do r[i] = b[i] - r[i];
        
        const bnorm = norm2(b);
        var rnorm = norm2(r);
        const rnorm0 = rnorm;
        
        // Check if already converged
        if rnorm < atol || (bnorm > 0.0 && rnorm / bnorm < rtol) {
            return (0, GMRESConvergedReason.Converged_ATOL);
        }
        
        // Outer restart loop
        while totalIter < maxIter {
            // RIGHT PRECONDITIONING: Initialize with TRUE residual
            // V[:,0] = r / ||r|| (no preconditioner applied here)
            
            if rnorm < 1e-14 {
                reason = GMRESConvergedReason.Diverged_Breakdown;
                break;
            }
            
            // Initialize Krylov basis: V[:,0] = r / ||r||
            const invRnorm = 1.0 / rnorm;
            forall i in 0..#n do V[i, 0] = r[i] * invRnorm;
            
            // Initialize g = [||r||, 0, 0, ...]
            g[0] = rnorm;
            for j in 1..restart do g[j] = 0.0;
            
            // Arnoldi iteration
            var converged = false;
            var j = 0;
            
            while j < restart && totalIter < maxIter {
                totalIter += 1;
                
                // RIGHT PRECONDITIONING: w = A * M^{-1} * V[:,j]
                // Step 1: z = M^{-1} * V[:,j]
                forall i in 0..#n do w[i] = V[i, j];
                applyPreconditioner(z, w);
                
                // Step 2: w = A * z
                A.matvec(w, z);
                
                // Modified Gram-Schmidt orthogonalization
                for i in 0..j {
                    var hij = 0.0;
                    forall k in 0..#n with (+ reduce hij) {
                        hij += w[k] * V[k, i];
                    }
                    H[i, j] = hij;
                    
                    forall k in 0..#n {
                        w[k] -= hij * V[k, i];
                    }
                }
                
                // H[j+1, j] = ||w||
                const wnorm = norm2(w);
                H[j+1, j] = wnorm;
                
                // Check for breakdown
                if wnorm < 1e-14 {
                    reason = GMRESConvergedReason.Diverged_Breakdown;
                    converged = true;
                    break;
                }
                
                // V[:,j+1] = w / ||w||
                const invWnorm = 1.0 / wnorm;
                forall i in 0..#n do V[i, j+1] = w[i] * invWnorm;
                
                // Apply previous Givens rotations to H[:,j]
                for i in 0..j-1 {
                    applyGivensRotation(H[i, j], H[i+1, j], cs[i], sn[i]);
                }
                
                // Generate new Givens rotation
                generateGivensRotation(H[j, j], H[j+1, j], cs[j], sn[j]);
                
                // Apply new rotation
                applyGivensRotation(H[j, j], H[j+1, j], cs[j], sn[j]);
                applyGivensRotation(g[j], g[j+1], cs[j], sn[j]);
                
                // Check convergence: |g[j+1]| is the residual norm
                rnorm = abs(g[j+1]);
                
                if rnorm < atol {
                    reason = GMRESConvergedReason.Converged_ATOL;
                    converged = true;
                    j += 1;
                    break;
                }
                if bnorm > 0.0 && rnorm / bnorm < rtol {
                    reason = GMRESConvergedReason.Converged_RTOL;
                    converged = true;
                    j += 1;
                    break;
                }
                if rnorm0 > 0.0 && rnorm / rnorm0 < rtol {
                    reason = GMRESConvergedReason.Converged_RTOL;
                    converged = true;
                    j += 1;
                    break;
                }
                
                j += 1;
            }
            
            // Solve upper triangular system H * y = g
            const m = j;  // Number of Arnoldi steps completed
            for i in (0..m-1 by -1) {
                y[i] = g[i];
                for k in i+1..m-1 {
                    y[i] -= H[i, k] * y[k];
                }
                y[i] /= H[i, i];
            }
            
            // RIGHT PRECONDITIONING: x = x + M^{-1} * (V[:,0:m-1] * y)
            // Step 1: Compute z = V * y (correction in Krylov space)
            forall i in 0..#n {
                var sum = 0.0;
                for k in 0..m-1 {
                    sum += V[i, k] * y[k];
                }
                z[i] = sum;
            }
            
            // Step 2: Apply preconditioner: w = M^{-1} * z
            applyPreconditioner(w, z);
            
            // Step 3: Update solution: x = x + w
            forall i in 0..#n {
                x[i] += w[i];
            }
            
            if converged then break;
            
            // Compute new residual for restart: r = b - A * x
            A.matvec(r, x);
            forall i in 0..#n do r[i] = b[i] - r[i];
            rnorm = norm2(r);
            
            // Check convergence after restart
            if rnorm < atol {
                reason = GMRESConvergedReason.Converged_ATOL;
                break;
            }
            if bnorm > 0.0 && rnorm / bnorm < rtol {
                reason = GMRESConvergedReason.Converged_RTOL;
                break;
            }
        }
        
        if totalIter >= maxIter && reason == GMRESConvergedReason.NotConverged {
            reason = GMRESConvergedReason.Diverged_MaxIt;
        }
        
        return (totalIter, reason);
    }
    
    // ========================================================================
    // Matrix-Free GMRES Solve (Newton-Krylov approach)
    //
    // Uses finite differences to compute Jacobian-vector products:
    //
    //   J * v ≈ (R(phi + h*v) - R(phi)) / h
    //
    // This is the Newton-Krylov approach from Blazek's CFD book (Eq. 6.62).
    //
    // jvpProvider should be an object with method:
    //   proc ref apply(ref result: [?D1] real(64), const ref v: [?D2] real(64))
    // that computes: result = J * v
    // ========================================================================
    proc ref solveMatrixFreeRight(
        const ref A_precon: SparseMatrixCSR,  // Matrix for preconditioner only
        ref x: [] real(64), 
        const ref b: [] real(64),
        ref jvpProvider,                       // Object with apply(result, v) method
        updatePrecon: bool = true             // Whether to update preconditioner
    ): (int, GMRESConvergedReason) {
        
        var totalIter = 0;
        var reason = GMRESConvergedReason.NotConverged;
        
        // For matrix-free, the initial residual is -R0 (we're solving J*dx = -R)
        // But we receive b = -R0, so r = b - J*x = b (since x starts at 0)
        forall i in 0..#n do r[i] = b[i];
        
        const bnorm = norm2(b);
        var rnorm = norm2(r);
        const rnorm0 = rnorm;
        
        // Check if already converged
        if rnorm < atol || (bnorm > 0.0 && rnorm / bnorm < rtol) {
            return (0, GMRESConvergedReason.Converged_ATOL);
        }
        
        // Setup preconditioner using the analytical Jacobian matrix (only if needed)
        if updatePrecon {
            if preconInitialized {
                updatePreconditioner(A_precon);
            } else {
                setupPreconditioner(A_precon);
            }
        } else if !preconInitialized {
            setupPreconditioner(A_precon);
        }
        
        // Outer restart loop
        while totalIter < maxIter {
            if rnorm < 1e-14 {
                reason = GMRESConvergedReason.Diverged_Breakdown;
                break;
            }
            
            // Initialize Krylov basis: V[:,0] = r / ||r||
            const invRnorm = 1.0 / rnorm;
            forall i in 0..#n do V[i, 0] = r[i] * invRnorm;
            
            // Initialize g = [||r||, 0, 0, ...]
            g[0] = rnorm;
            for j in 1..restart do g[j] = 0.0;
            
            // Arnoldi iteration
            var converged = false;
            var j = 0;
            
            while j < restart && totalIter < maxIter {
                totalIter += 1;
                
                // RIGHT PRECONDITIONING: w = J * M^{-1} * V[:,j]
                // Step 1: z = M^{-1} * V[:,j]
                forall i in 0..#n do w[i] = V[i, j];
                applyPreconditioner(z, w);
                
                // Step 2: w = J * z (using matrix-free Jacobian-vector product)
                jvpProvider.apply(w, z);
                
                // Modified Gram-Schmidt orthogonalization
                for i in 0..j {
                    var hij = 0.0;
                    forall k in 0..#n with (+ reduce hij) {
                        hij += w[k] * V[k, i];
                    }
                    H[i, j] = hij;
                    
                    forall k in 0..#n {
                        w[k] -= hij * V[k, i];
                    }
                }
                
                // H[j+1, j] = ||w||
                const wnorm = norm2(w);
                H[j+1, j] = wnorm;
                
                // Check for breakdown
                if wnorm < 1e-14 {
                    reason = GMRESConvergedReason.Diverged_Breakdown;
                    converged = true;
                    break;
                }
                
                // V[:,j+1] = w / ||w||
                const invWnorm = 1.0 / wnorm;
                forall i in 0..#n do V[i, j+1] = w[i] * invWnorm;
                
                // Apply previous Givens rotations to H[:,j]
                for i in 0..j-1 {
                    applyGivensRotation(H[i, j], H[i+1, j], cs[i], sn[i]);
                }
                
                // Generate new Givens rotation
                generateGivensRotation(H[j, j], H[j+1, j], cs[j], sn[j]);
                
                // Apply new rotation
                applyGivensRotation(H[j, j], H[j+1, j], cs[j], sn[j]);
                applyGivensRotation(g[j], g[j+1], cs[j], sn[j]);
                
                // Check convergence: |g[j+1]| is the residual norm
                rnorm = abs(g[j+1]);
                
                if rnorm < atol {
                    reason = GMRESConvergedReason.Converged_ATOL;
                    converged = true;
                    j += 1;
                    break;
                }
                if bnorm > 0.0 && rnorm / bnorm < rtol {
                    reason = GMRESConvergedReason.Converged_RTOL;
                    converged = true;
                    j += 1;
                    break;
                }
                if rnorm0 > 0.0 && rnorm / rnorm0 < rtol {
                    reason = GMRESConvergedReason.Converged_RTOL;
                    converged = true;
                    j += 1;
                    break;
                }
                
                j += 1;
            }
            
            // Solve upper triangular system H * y = g
            const m = j;
            for i in (0..m-1 by -1) {
                y[i] = g[i];
                for k in i+1..m-1 {
                    y[i] -= H[i, k] * y[k];
                }
                y[i] /= H[i, i];
            }
            
            // RIGHT PRECONDITIONING: x = x + M^{-1} * (V * y)
            forall i in 0..#n {
                var sum = 0.0;
                for k in 0..m-1 {
                    sum += V[i, k] * y[k];
                }
                z[i] = sum;
            }
            
            applyPreconditioner(w, z);
            
            forall i in 0..#n {
                x[i] += w[i];
            }
            
            if converged then break;
            
            // Compute new residual for restart using matrix-free product
            // r = b - J*x
            jvpProvider.apply(w, x);
            forall i in 0..#n do r[i] = b[i] - w[i];
            rnorm = norm2(r);
            
            if rnorm < atol {
                reason = GMRESConvergedReason.Converged_ATOL;
                break;
            }
            if bnorm > 0.0 && rnorm / bnorm < rtol {
                reason = GMRESConvergedReason.Converged_RTOL;
                break;
            }
        }
        
        if totalIter >= maxIter && reason == GMRESConvergedReason.NotConverged {
            reason = GMRESConvergedReason.Diverged_MaxIt;
        }
        
        return (totalIter, reason);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

} // module gmres

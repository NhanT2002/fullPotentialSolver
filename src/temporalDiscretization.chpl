module temporalDiscretization
{
use mesh;
use writeCGNS;
use Math;
use linearAlgebra;
use Time;
import input.potentialInputs;
use spatialDiscretization;
use PETSCapi;
use C_PETSC;
use petsc;
use CTypes;
use List;
use gmres;
use Sort;

// ============================================================================
// Jacobian-Vector Product Provider for Matrix-Free Newton-Krylov
// Uses finite differences: Jv = (R(phi + h*v) - R(phi)) / h (Blazek Eq. 6.62)
// ============================================================================
class JacobianVectorProductProvider {
    var spatialDisc_: shared spatialDiscretization;
    var n_: int;  // Number of domain elements
    
    // State for finite difference computation
    var phi_current_dom_: domain(1);
    var phi_current_: [phi_current_dom_] real(64);
    var gamma_current_: real(64);
    var R0_: [phi_current_dom_] real(64);
    var kutta_res0_: real(64);
    
    // Temporary storage for perturbed residual
    var phi_backup_dom_: domain(1);
    var phi_backup_: [phi_backup_dom_] real(64);
    
    // Backup arrays for intermediate quantities (avoid recomputing run() after restore)
    var uu_backup_: [phi_backup_dom_] real(64);
    var vv_backup_: [phi_backup_dom_] real(64);
    var rhorho_backup_: [phi_backup_dom_] real(64);
    var res_backup_: [phi_backup_dom_] real(64);
    
    proc init(spatialDisc: shared spatialDiscretization) {
        this.spatialDisc_ = spatialDisc;
        this.n_ = spatialDisc.nelemDomain_;
        this.phi_current_dom_ = {1..this.n_};
        this.phi_backup_dom_ = {1..this.n_};
    }
    
    // Store the current state for finite difference computation
    proc ref setState(const ref phi: [] real(64), gamma: real(64),
                      const ref R0: [] real(64), kutta_res0: real(64)) {
        forall i in 1..n_ {
            this.phi_current_[i] = phi[i];
            this.R0_[i] = R0[i-1];  // R0 is 0-indexed from GMRES
        }
        this.gamma_current_ = gamma;
        this.kutta_res0_ = kutta_res0;
    }
    
    // Apply the Jacobian-vector product: result = J * v
    // Uses finite differences: Jv = (R(phi + h*v) - R(phi)) / h
    proc ref apply(ref result: [] real(64), const ref v: [] real(64)) {
        // Blazek Eq. 6.63: Step size calculation
        // h = sqrt(eps) * max(|d|, typ_W * ||W||) * sign(d) / ||v||
        const eps = 1.0e-14;  // Machine epsilon for float64
        const sqrtEps = sqrt(eps);
        const typ_W = 1.0;  // Typical magnitude of solution components
        
        // Compute ||v||
        var vnorm = 0.0;
        forall i in 0..#n_ with (+ reduce vnorm) {
            vnorm += v[i] * v[i];
        }
        // Include circulation component
        vnorm += v[n_] * v[n_];
        vnorm = sqrt(vnorm);
        
        // Avoid division by zero
        if vnorm < 1.0e-30 {
            forall i in 0..#(n_+1) do result[i] = 0.0;
            return;
        }
        
        // Compute d = phi · v (dot product with solution)
        var d = 0.0;
        forall i in 1..n_ with (+ reduce d) {
            d += phi_current_[i] * v[i-1];  // v is 0-indexed
        }
        // Include circulation contribution
        d += gamma_current_ * v[n_];
        
        // Compute ||phi|| for scaling
        var phinorm = 0.0;
        forall i in 1..n_ with (+ reduce phinorm) {
            phinorm += phi_current_[i] * phi_current_[i];
        }
        phinorm += gamma_current_ * gamma_current_;
        phinorm = sqrt(phinorm);
        
        // Step size (Blazek Eq. 6.63)
        // Use a smaller, more conservative step size for better accuracy
        var h = 1.0e-8 * max(1.0, phinorm);
        
        // Save current spatial discretization state (phi, circulation, and derived quantities)
        forall i in 1..n_ {
            phi_backup_[i] = this.spatialDisc_.phi_[i];
            uu_backup_[i] = this.spatialDisc_.uu_[i];
            vv_backup_[i] = this.spatialDisc_.vv_[i];
            rhorho_backup_[i] = this.spatialDisc_.rhorho_[i];
            res_backup_[i] = this.spatialDisc_.res_[i];
        }
        const gamma_backup = this.spatialDisc_.circulation_;
        const kutta_res_backup = this.spatialDisc_.kutta_res_;
        
        // Perturb solution: phi = phi_current + h * v
        forall i in 1..n_ {
            this.spatialDisc_.phi_[i] = phi_current_[i] + h * v[i-1];
        }
        this.spatialDisc_.circulation_ = gamma_current_ + h * v[n_];
        
        // Compute perturbed residual R(phi + h*v) using lightweight run_jv
        this.spatialDisc_.run_jv();
        
        // Finite difference approximation: Jv = (R_perturbed - R0) / h
        const invH = 1.0 / h;
        forall i in 0..#n_ {
            result[i] = (this.spatialDisc_.res_[i+1] - R0_[i+1]) * invH;
        }
        result[n_] = (this.spatialDisc_.kutta_res_ - kutta_res0_) * invH;
        
        // Restore original spatial discretization state (no need to call run() again!)
        forall i in 1..n_ {
            this.spatialDisc_.phi_[i] = phi_backup_[i];
            this.spatialDisc_.uu_[i] = uu_backup_[i];
            this.spatialDisc_.vv_[i] = vv_backup_[i];
            this.spatialDisc_.rhorho_[i] = rhorho_backup_[i];
            this.spatialDisc_.res_[i] = res_backup_[i];
        }
        this.spatialDisc_.circulation_ = gamma_backup;
        this.spatialDisc_.kutta_res_ = kutta_res_backup;
    }
}

class temporalDiscretization {
    var spatialDisc_: shared spatialDiscretization;
    var inputs_: potentialInputs;
    var it_: int = 0;
    var t0_: real(64) = 0.0;
    var first_res_: real(64) = 1e12;
    
    // Index for circulation DOF (last row/column) - must be before A_petsc for init order
    var gammaIndex_: int;
    
    var A_petsc : owned PETSCmatrix_c;
    var x_petsc : owned PETSCvector_c;
    var b_petsc : owned PETSCvector_c;
    var ksp : owned PETSCksp_c;

    // Native GMRES solver components
    var nativeGmres_: GMRESSolver;
    
    // Matrix-free Jacobian-vector product provider (must come after nativeGmres_ for init order)
    var jvpProvider_: owned JacobianVectorProductProvider?;
    
    var A_csr_: SparseMatrixCSR;
    var x_native_dom_: domain(1) = {0..#1};
    var x_native_: [x_native_dom_] real(64);
    var b_native_: [x_native_dom_] real(64);

    var timeList_ = new list(real(64));
    var itList_ = new list(int);
    var resList_ = new list(real(64));
    var clList_ = new list(real(64));
    var cdList_ = new list(real(64));
    var cmList_ = new list(real(64));
    var circulationList_ = new list(real(64));

    // Gradient sensitivity to circulation: ∂(∇φ)/∂Γ for each cell
    // These capture how each cell's gradient depends on Γ through wake-crossing faces
    var gradSensitivity_dom: domain(1) = {1..0};
    var dgradX_dGamma_: [gradSensitivity_dom] real(64);
    var dgradY_dGamma_: [gradSensitivity_dom] real(64);
    var Jij_: [gradSensitivity_dom] real(64);

    // Selective Frequency Damping (SFD) - Jordi et al. 2014 encapsulated formulation
    // Filters low-frequency oscillations to accelerate convergence
    // φ_bar is the temporally filtered solution
    var phi_bar_: [gradSensitivity_dom] real(64);
    var circulation_bar_: real(64) = 0.0;

    proc init(spatialDisc: shared spatialDiscretization, ref inputs: potentialInputs) {
        writeln("Initializing temporal discretization...");
        this.spatialDisc_ = spatialDisc;
        this.inputs_ = inputs;

        // Add 1 extra DOF for circulation Γ
        const M = spatialDisc.nelemDomain_ + 1;
        const N = spatialDisc.nelemDomain_ + 1;
        this.gammaIndex_ = spatialDisc.nelemDomain_;  // 0-based index for Γ
        
        this.A_petsc = new owned PETSCmatrix_c(PETSC_COMM_SELF, "seqaij", M, M, N, N);
        this.x_petsc = new owned PETSCvector_c(PETSC_COMM_SELF, N, N, 0.0, "seq");
        this.b_petsc = new owned PETSCvector_c(PETSC_COMM_SELF, N, N, 0.0, "seq");

        var nnz : [0..M-1] PetscInt;
        nnz = 4*(this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[1] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[1 + 1]].size + 1);
        // Γ row has connections to TE cells only
        nnz[M-1] = 5;
        A_petsc.preAllocate(nnz);

        this.ksp = new owned PETSCksp_c(PETSC_COMM_SELF, "gmres");
        this.ksp.setTolerances(inputs.GMRES_RTOL_, inputs.GMRES_ATOL_, inputs.GMRES_DTOL_, inputs.GMRES_MAXIT_);
        this.ksp.GMRESSetRestart(inputs.GMRES_RESTART_);
        this.ksp.GMRESSetPreAllocateVectors();
        if this.inputs_.GMRES_PRECON_ == "jacobi" {
            writeln("Using Jacobi preconditioner for GMRES");
            this.ksp.setPreconditioner("jacobi");
        } else if this.inputs_.GMRES_PRECON_ == "ilu" {
            writeln("Using ILU preconditioner for GMRES");
            this.ksp.setPreconditioner("ilu");
        } else if this.inputs_.GMRES_PRECON_ == "lu" {
            writeln("Using lu preconditioner for GMRES");
            this.ksp.setPreconditioner("lu");
        } else if this.inputs_.GMRES_PRECON_ == "asm" {
            writeln("Using asm preconditioner for GMRES");
            this.ksp.setPreconditioner("asm");
        } else if this.inputs_.GMRES_PRECON_ == "gasm" {
            writeln("Using gasm preconditioner for GMRES");
            this.ksp.setPreconditioner("gasm");
        } else if this.inputs_.GMRES_PRECON_ == "bjacobi" {
            writeln("Using bjacobi preconditioner for GMRES");
            this.ksp.setPreconditioner("bjacobi");
        } else if this.inputs_.GMRES_PRECON_ == "none" {
            writeln("Using no preconditioner for GMRES");
            this.ksp.setPreconditioner("none");
        } else {
            writeln("No preconditioner for GMRES");
            this.ksp.setPreconditioner("none");
        }

        // Initialize native GMRES arrays (must come before gradSensitivity_dom for field order)
        if this.inputs_.USE_NATIVE_GMRES_ {
            writeln("Using native Chapel GMRES solver");
            this.x_native_dom_ = {0..#N};
        }
        
        // Initialize gradient sensitivity arrays
        this.gradSensitivity_dom = {1..spatialDisc.nelemDomain_};

        // Initialize native GMRES solver (after other fields)
        if this.inputs_.USE_NATIVE_GMRES_ {
            // Determine preconditioner type
            var preconType = PreconditionerType.None;
            if this.inputs_.GMRES_PRECON_ == "jacobi" {
                writeln("Using Jacobi preconditioner for native GMRES");
                preconType = PreconditionerType.Jacobi;
            } else if this.inputs_.GMRES_PRECON_ == "ilu" {
                writeln("Using ILU(0) preconditioner for native GMRES");
                preconType = PreconditionerType.ILU0;
            } else {
                writeln("Using no preconditioner for native GMRES");
            }
            
            // Determine preconditioning side
            var preconSide = PreconSide.Right;
            if this.inputs_.GMRES_PRECON_SIDE_ == "left" {
                writeln("Using LEFT preconditioning (preconditioned residual)");
                preconSide = PreconSide.Left;
            } else {
                writeln("Using RIGHT preconditioning (true residual)");
                preconSide = PreconSide.Right;
            }
            
            // Print Jacobian type
            if this.inputs_.JACOBIAN_TYPE_ == "numerical" {
                writeln("Using NUMERICAL (matrix-free) Jacobian via finite differences (Blazek Eq. 6.62)");
            } else {
                writeln("Using ANALYTICAL Jacobian");
            }
            
            this.nativeGmres_ = new GMRESSolver(
                N, 
                inputs.GMRES_RESTART_,
                inputs.GMRES_MAXIT_,
                inputs.GMRES_RTOL_,
                inputs.GMRES_ATOL_,
                preconType,
                preconSide
            );
            
            // Initialize JVP provider if numerical Jacobian (after nativeGmres_ for field order)
            if this.inputs_.JACOBIAN_TYPE_ == "numerical" {
                this.jvpProvider_ = new owned JacobianVectorProductProvider(spatialDisc);
            }
        }

        // Print SFD status
        if this.inputs_.SFD_ENABLED_ {
            writeln("SFD enabled: χ = ", this.inputs_.SFD_CHI_, ", Δ = ", this.inputs_.SFD_DELTA_);
        }
    }

    proc initializeJacobian() {
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            this.A_petsc.set(elem-1, elem-1, 0.0);
            const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
            for face in faces {
                const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;
                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    this.A_petsc.set(elem-1, neighbor-1, 0.0);
                }
            }
            // Initialize dRes_i/dΓ column entries for wake-adjacent cells
            this.A_petsc.set(elem-1, this.gammaIndex_, 0.0);
        }
        
        // Initialize Γ row (Kutta condition): Γ - (φ_upper - φ_lower) = 0
        // dKutta/dΓ = 1
        this.A_petsc.set(this.gammaIndex_, this.gammaIndex_, 0.0);
        // dKutta/dφ_upperTE = -1
        this.A_petsc.set(this.gammaIndex_, this.spatialDisc_.upperTEelem_ - 1, 0.0);
        // dKutta/dφ_lowerTE = +1
        this.A_petsc.set(this.gammaIndex_, this.spatialDisc_.lowerTEelem_ - 1, 0.0);

        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
    }

    proc computeGradientSensitivity() {
        // Compute ∂(∇φ)/∂Γ for each cell.
        // This captures how each cell's gradient depends on circulation through
        // its wake-crossing faces.
        //
        // For cell I with gradient: ∇φ_I = Σ_k w_Ik * (φ_k_corrected - φ_I)
        // where φ_k_corrected includes the Γ correction for wake-crossing neighbors:
        //   - If I above (1), k below (-1): φ_k_corrected = φ_k + Γ
        //   - If I below (-1), k above (1): φ_k_corrected = φ_k - Γ
        //
        // Therefore: ∂(∇φ_I)/∂Γ = Σ_{k: wake-crossing} w_Ik * (±1)

        // Reset arrays
        this.dgradX_dGamma_ = 0.0;
        this.dgradY_dGamma_ = 0.0;

        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            const kuttaType_elem = this.spatialDisc_.kuttaCell_[elem];

            // Only cells in wake region (above=1 or below=-1) can have wake-crossing faces
            if kuttaType_elem == 1 || kuttaType_elem == -1 {
                const faces = this.spatialDisc_.mesh_.elem2edge_[
                    this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 ..
                    this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];

                var dgx = 0.0, dgy = 0.0;

                for face in faces {
                    const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                    const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                    const neighbor = if elem1 == elem then elem2 else elem1;

                    const kuttaType_neighbor = this.spatialDisc_.kuttaCell_[neighbor];

                    // Check if this is a wake-crossing face
                    const isWakeCrossing = (kuttaType_elem == 1 && kuttaType_neighbor == -1) ||
                                           (kuttaType_elem == -1 && kuttaType_neighbor == 1);

                    if isWakeCrossing {
                        // Get weight from elem to neighbor
                        var wx, wy: real(64);
                        if elem == elem1 {
                            wx = this.spatialDisc_.lsGradQR_!.wxFinal1_[face];
                            wy = this.spatialDisc_.lsGradQR_!.wyFinal1_[face];
                        } else {
                            wx = this.spatialDisc_.lsGradQR_!.wxFinal2_[face];
                            wy = this.spatialDisc_.lsGradQR_!.wyFinal2_[face];
                        }

                        // Sign: +1 if elem above, neighbor below (φ_neighbor + Γ)
                        //       -1 if elem below, neighbor above (φ_neighbor - Γ)
                        const gammaSgn = if kuttaType_elem == 1 then 1.0 else -1.0;

                        dgx += wx * gammaSgn;
                        dgy += wy * gammaSgn;
                    }
                }

                this.dgradX_dGamma_[elem] = dgx;
                this.dgradY_dGamma_[elem] = dgy;
            }
        }
    }

    proc computeJacobian() {
        // Compute the Jacobian matrix d(res_I)/d(phi_J) for the linear system.
        //
        // Residual: res_I = sum over faces f of elem I: sign_f * flux_f
        //
        // Flux with deferred correction (from spatialDiscretization.computeFluxes):
        //   V_face = V_avg - delta * corrCoeff
        //   where delta = V_avg · t - (phi_2 - phi_1) * invL
        //   and corrCoeff = n / (n · t)
        //
        // Expanding V_face · n:
        //   V_face · n = V_avg · n - delta * (corrCoeff · n)
        //              = V_avg · n - (V_avg · t - dPhi/dL) * k
        //   where k = 1/(n · t) = corrCoeff · n
        //
        //   = V_avg · (n - k*t) + k * invL * (phi_2 - phi_1)
        //   = 0.5*(gradPhi_1 + gradPhi_2) · m + directCoeff * (phi_2 - phi_1)
        //
        // where m = n - k*t is the effective normal, and directCoeff = k * invL
        //
        // For WALL boundaries:
        //   phi_ghost = phi_interior (Neumann BC)
        //   V_ghost = V_int - 2*(V_int·n)*n (mirror velocity)
        //   V_avg = V_int - (V_int·n)*n = V_int,tangent
        //
        // So V_avg · m = V_int · m - (V_int·n)*(n·m)
        //              = V_int · [m - (n·m)*n]
        // Define m_wall = m - (n·m)*n (tangential projection of effective normal)
        //
        // And the direct term: phi_ghost - phi_int = 0
        //
        // Derivatives:
        //   gradPhi_I = sum_k w_Ik * (phi_k - phi_I)
        //   d(gradPhi_I)/d(phi_I) = -sumW_I
        //   d(gradPhi_I)/d(phi_k) = w_Ik

        this.A_petsc.zeroEntries();
        
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            const faces = this.spatialDisc_.mesh_.elem2edge_[
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. 
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
            
            var diag = 0.0;  // Diagonal contribution d(res_elem)/d(phi_elem)
            var dRes_dGamma = 0.0;  // Contribution d(res_elem)/d(Γ) for wake-crossing cells
            
            for face in faces {
                const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;
                
                // Sign: +1 if elem is elem1 (flux outward), -1 if elem is elem2
                const sign = if elem1 == elem then 1.0 else -1.0;
                
                // Face geometry
                const nx = this.spatialDisc_.faceNormalX_[face];
                const ny = this.spatialDisc_.faceNormalY_[face];
                const area = this.spatialDisc_.faceArea_[face];
                const rhoFace = this.spatialDisc_.rhoFace_[face];
                
                // Get precomputed correction coefficients
                const t_x = this.spatialDisc_.t_IJ_x_[face];
                const t_y = this.spatialDisc_.t_IJ_y_[face];
                const invL = this.spatialDisc_.invL_IJ_[face];
                const nDotT = nx * t_x + ny * t_y;
                const k = 1.0 / nDotT;
                
                // Effective normal: m = n - k*t (accounts for deferred correction)
                const mx = nx - k * t_x;
                const my = ny - k * t_y;
                
                // Direct phi coefficient: k * invL
                const directCoeff = k * invL;
                
                // Get gradient weights for this element
                const sumWx_elem = this.spatialDisc_.lsGradQR_!.sumWx_[elem];
                const sumWy_elem = this.spatialDisc_.lsGradQR_!.sumWy_[elem];
                var wx_elemToNeighbor: real(64);
                var wy_elemToNeighbor: real(64);
                var wx_neighborToElem: real(64);
                var wy_neighborToElem: real(64);
                
                if elem1 == elem {
                    // elem is elem1, using weights from perspective 1
                    wx_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wxFinal1_[face];
                    wy_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wyFinal1_[face];
                    wx_neighborToElem = this.spatialDisc_.lsGradQR_!.wxFinal2_[face];
                    wy_neighborToElem = this.spatialDisc_.lsGradQR_!.wyFinal2_[face];
                } else {
                    // elem is elem2, using weights from perspective 2
                    wx_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wxFinal2_[face];
                    wy_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wyFinal2_[face];
                    wx_neighborToElem = this.spatialDisc_.lsGradQR_!.wxFinal1_[face];
                    wy_neighborToElem = this.spatialDisc_.lsGradQR_!.wyFinal1_[face];
                }
                
                // Check if this is a boundary face (neighbor is ghost cell)
                const isInteriorFace = neighbor <= this.spatialDisc_.nelemDomain_;
                const isWallFace = this.spatialDisc_.wallFaceSet_.contains(face);
                
                // Check if this face crosses the wake (Kutta condition)
                // kuttaCell_ = 1 (above wake), -1 (below wake), 9 (elsewhere)
                const kuttaType_elem = this.spatialDisc_.kuttaCell_[elem];
                const kuttaType_neighbor = this.spatialDisc_.kuttaCell_[neighbor];
                const isWakeCrossingFace = (kuttaType_elem == 1 && kuttaType_neighbor == -1) ||
                                           (kuttaType_elem == -1 && kuttaType_neighbor == 1);
                
                if isInteriorFace {
                    // === INTERIOR FACE ===
                    // V_avg = 0.5*(V_elem + V_neighbor)
                    // flux·n = 0.5*(gradPhi_elem + gradPhi_neighbor)·m + directCoeff*(phi_neighbor - phi_elem)
                    
                    // === DIAGONAL CONTRIBUTION ===
                    // From d(0.5*(gradPhi_elem · m))/d(phi_elem) = 0.5 * (-sumW_elem · m)
                    var face_diag = 0.5 * (-sumWx_elem * mx - sumWy_elem * my);
                    
                    // From d(0.5*(gradPhi_neighbor · m))/d(phi_elem) = 0.5 * (w_neighborToElem · m)
                    face_diag += 0.5 * (wx_neighborToElem * mx + wy_neighborToElem * my);
                    
                    // Apply sign and area to gradient terms
                    diag += sign * face_diag * area * rhoFace;
                    
                    // Direct phi term: d((phi_2 - phi_1) * directCoeff)/d(phi_elem)
                    diag -= directCoeff * area * rhoFace;
                    
                    // === DENSITY RETARDATION FOR TRANSONIC FLOWS ===
                    // In supersonic regions, ρ_face depends on upwind density:
                    //   ρ_face = (1-μ) * ρ_isen + μ * ρ_upwind
                    // The flux F = ρ_face * (V·n) * A includes ∂ρ_face/∂φ terms.
                    //
                    // Full derivative:
                    // ∂ρ_face/∂φ = (1-μ)*∂ρ_isen/∂φ + μ*∂ρ_upwind/∂φ + (∂μ/∂φ)*(ρ_upwind - ρ_isen)
                    //
                    // ∂F/∂φ = ρ_face * ∂(V·n)/∂φ * A + (V·n) * ∂ρ_face/∂φ * A
                    //                 ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^
                    //                 (already included)    (add these terms)
                    //
                    // Compute face velocity and normal velocity
                    const uFace = this.spatialDisc_.uFace_[face];
                    const vFace = this.spatialDisc_.vFace_[face];
                    const vDotN = uFace * nx + vFace * ny;
                    
                    // Determine upwind element
                    const upwindElem = this.spatialDisc_.upwindElem_[face];
                    
                    // Compute local Mach at upwind cell
                    const rhoUpwind = this.spatialDisc_.rhorho_[upwindElem];
                    const uUpwind = this.spatialDisc_.uu_[upwindElem];
                    const vUpwind = this.spatialDisc_.vv_[upwindElem];
                    const machUpwind = this.spatialDisc_.machmach_[upwindElem];
                    
                    // Switching function: μ = μ_c * max(0, M² - M_c²)
                    const mu_upwind = this.spatialDisc_.mumu_[upwindElem];
                    
                    // Add density retardation if in supersonic region
                    if mu_upwind > 0.0 {
                        const gamma = this.inputs_.GAMMA_;
                        const Minf2 = this.inputs_.MACH_ * this.inputs_.MACH_;
                        const velMag2 = uUpwind*uUpwind + vUpwind*vUpwind;
                        const velMag = sqrt(velMag2);
                        
                        // Get isentropic density at face for ∂μ/∂φ term
                        // ρ_isen was the original rhoFace before artificial density was applied
                        // We can recover it: ρ_face = ρ_isen - μ*(ρ_isen - ρ_upwind)
                        // => ρ_isen = (ρ_face - μ*ρ_upwind) / (1 - μ)
                        const rhoIsen = if mu_upwind < 0.999 
                                        then (rhoFace - mu_upwind * rhoUpwind) / (1.0 - mu_upwind)
                                        else rhoUpwind;
                        
                        // === Term 1: μ * ∂ρ_upwind/∂φ (already implemented) ===
                        // ∂ρ_upwind/∂V = -ρ^(2-γ) * M∞² * |V|
                        const drho_dVelMag = -rhoUpwind**(2.0-gamma) * Minf2 * velMag;
                        
                        // ∂|V|/∂φ_upwind = -sumW · (V/|V|)
                        var dVelMag_dPhi_upwind = 0.0;
                        if velMag > 1e-10 {
                            const sumWx_upwind = this.spatialDisc_.lsGradQR_!.sumWx_[upwindElem];
                            const sumWy_upwind = this.spatialDisc_.lsGradQR_!.sumWy_[upwindElem];
                            dVelMag_dPhi_upwind = (-sumWx_upwind * uUpwind - sumWy_upwind * vUpwind) / velMag;
                        }
                        
                        // ∂ρ_face/∂φ from Term 1: μ * ∂ρ_upwind/∂φ
                        var drhoFace_dPhi_upwind = mu_upwind * drho_dVelMag * dVelMag_dPhi_upwind;
                        
                        // === Term 2: (∂μ/∂φ) * (ρ_upwind - ρ_isen) ===
                        // μ = μ_c * (M² - M_c²)
                        // ∂μ/∂M = μ_c * 2*M
                        // ∂M/∂|V| ≈ M / |V| (dominant term, treating ρ as approximately constant)
                        // ∂μ/∂|V| = ∂μ/∂M * ∂M/∂|V| = μ_c * 2*M * (M/|V|) = μ_c * 2*M²/|V|
                        // ∂μ/∂φ = ∂μ/∂|V| * ∂|V|/∂φ
                        if velMag > 1e-10 {
                            const dmu_dVelMag = this.inputs_.MU_C_ * 2.0 * machUpwind * machUpwind / velMag;
                            const dmu_dPhi_upwind = dmu_dVelMag * dVelMag_dPhi_upwind;
                            
                            // Add contribution: (∂μ/∂φ) * (ρ_upwind - ρ_isen)
                            drhoFace_dPhi_upwind += dmu_dPhi_upwind * (rhoUpwind - rhoIsen);
                        }
                        
                        // Add contribution to Jacobian: (V·n) * ∂ρ_face/∂φ_upwind * A
                        const densityRetardation = vDotN * drhoFace_dPhi_upwind * area;
                        
                        if upwindElem == elem {
                            // Add to diagonal
                            diag += sign * densityRetardation;
                        }
                        // Note: off-diagonal contribution for upwind neighbor is handled below
                    }
                    
                    // === OFF-DIAGONAL CONTRIBUTION ===
                    const sumWx_neighbor = this.spatialDisc_.lsGradQR_!.sumWx_[neighbor];
                    const sumWy_neighbor = this.spatialDisc_.lsGradQR_!.sumWy_[neighbor];
                    
                    // From d(0.5*(gradPhi_elem · m))/d(phi_neighbor) = 0.5 * (w_elemToNeighbor · m)
                    var offdiag = 0.5 * (wx_elemToNeighbor * mx + wy_elemToNeighbor * my);
                    
                    // From d(0.5*(gradPhi_neighbor · m))/d(phi_neighbor) = 0.5 * (-sumW_neighbor · m)
                    offdiag += 0.5 * (-sumWx_neighbor * mx - sumWy_neighbor * my);
                    
                    // Apply sign and area to gradient terms
                    offdiag *= sign * area * rhoFace;
                    
                    // Direct phi term
                    offdiag += directCoeff * area * rhoFace;
                    
                    // Add density retardation for off-diagonal if neighbor is upwind
                    if mu_upwind > 0.0 && upwindElem == neighbor {
                        const gamma = this.inputs_.GAMMA_;
                        const Minf2 = this.inputs_.MACH_ * this.inputs_.MACH_;
                        const velMag2 = uUpwind*uUpwind + vUpwind*vUpwind;
                        const velMag = sqrt(velMag2);
                        const drho_dVelMag = -rhoUpwind**(2.0-gamma) * Minf2 * velMag;
                        
                        // Get isentropic density at face
                        const rhoIsen = if mu_upwind < 0.999 
                                        then (rhoFace - mu_upwind * rhoUpwind) / (1.0 - mu_upwind)
                                        else rhoUpwind;
                        
                        var dVelMag_dPhi_neighbor = 0.0;
                        if velMag > 1e-10 {
                            dVelMag_dPhi_neighbor = (-sumWx_neighbor * uUpwind - sumWy_neighbor * vUpwind) / velMag;
                        }
                        
                        // Term 1: μ * ∂ρ_upwind/∂φ_neighbor
                        var drhoFace_dPhi_neighbor = mu_upwind * drho_dVelMag * dVelMag_dPhi_neighbor;
                        
                        // Term 2: (∂μ/∂φ_neighbor) * (ρ_upwind - ρ_isen)
                        if velMag > 1e-10 {
                            const dmu_dVelMag = this.inputs_.MU_C_ * 2.0 * machUpwind * machUpwind / velMag;
                            const dmu_dPhi_neighbor = dmu_dVelMag * dVelMag_dPhi_neighbor;
                            drhoFace_dPhi_neighbor += dmu_dPhi_neighbor * (rhoUpwind - rhoIsen);
                        }
                        
                        const densityRetardation = vDotN * drhoFace_dPhi_neighbor * area;
                        offdiag += sign * densityRetardation;
                    }
                    
                    this.A_petsc.add(elem-1, neighbor-1, offdiag * this.spatialDisc_.invElemVolume_[elem]);

                    // === CIRCULATION (Γ) DERIVATIVE ===
                    // flux = 0.5 * (∇φ_elem + ∇φ_neighbor) · m + directCoeff * (φ_neighbor - φ_elem)
                    // ∂flux/∂Γ = 0.5 * (∂∇φ_elem/∂Γ + ∂∇φ_neighbor/∂Γ) · m + ∂(direct)/∂Γ
                    //
                    // The gradient sensitivities are precomputed for ALL cells, capturing
                    // contributions from ALL their wake-crossing faces (not just this face).
                    // This ensures correct Jacobian even for faces that are not themselves
                    // wake-crossing but whose cells have wake-crossing neighbors.

                    // Contribution from elem's gradient sensitivity
                    const dgradX_elem = this.dgradX_dGamma_[elem];
                    const dgradY_elem = this.dgradY_dGamma_[elem];
                    var dFlux_dGamma = 0.5 * (dgradX_elem * mx + dgradY_elem * my);

                    // Contribution from neighbor's gradient sensitivity
                    const dgradX_neighbor = this.dgradX_dGamma_[neighbor];
                    const dgradY_neighbor = this.dgradY_dGamma_[neighbor];
                    dFlux_dGamma += 0.5 * (dgradX_neighbor * mx + dgradY_neighbor * my);

                    // Apply sign and area
                    dFlux_dGamma *= sign * area * rhoFace;

                    // Direct term: only for wake-crossing faces
                    // directCoeff * ((φ_neighbor ± Γ) - φ_elem)
                    // ∂/∂Γ = ±directCoeff (sign depends on which side of wake)
                    if isWakeCrossingFace {
                        var gammaSgn = 0.0;
                        if kuttaType_elem == 1 && kuttaType_neighbor == -1 {
                            gammaSgn = 1.0;  // elem above, neighbor below: +Γ
                        } else {
                            gammaSgn = -1.0; // elem below, neighbor above: -Γ
                        }
                        dFlux_dGamma += directCoeff * area * rhoFace * gammaSgn;
                    }

                    dRes_dGamma += dFlux_dGamma;

                } else if isWallFace {
                    // === WALL BOUNDARY FACE ===
                    // Wall BC: phi_ghost = phi_interior (Neumann)
                    //          V_ghost = V_int - 2*(V_int·n)*n (mirror velocity)
                    //
                    // This gives: V_avg = V_int - (V_int·n)*n (tangential projection)
                    // And: V_avg · m = V_int · m_wall, where m_wall = m - (n·m)*n
                    //
                    // The interior gradient includes ghost as neighbor:
                    //   gradPhi_int = sum_k w_ik * (phi_k - phi_int)
                    //   d(gradPhi_int)/d(phi_int) = -sumW_int + w_int_to_ghost * d(phi_ghost)/d(phi_int)
                    //                             = -sumW_int + w_elemToNeighbor * 1
                    //
                    // So the diagonal contribution is:
                    //   d(V_avg · m)/d(phi_int) = d(V_int · m_wall)/d(phi_int)
                    //                          = (-sumW_int + w_elemToNeighbor) · m_wall
                    
                    // Compute m_wall = m - (n·m)*n (tangential projection of effective normal)
                    const nDotM = nx * mx + ny * my;
                    const mWallX = mx - nDotM * nx;
                    const mWallY = my - nDotM * ny;
                    
                    // Diagonal contribution from d(gradPhi_elem)/d(phi_elem)
                    // Note: includes correction for d(phi_ghost)/d(phi_int) = 1
                    var face_diag = (-sumWx_elem + wx_elemToNeighbor) * mWallX 
                                  + (-sumWy_elem + wy_elemToNeighbor) * mWallY;
                    
                    // Apply sign and area
                    diag += sign * face_diag * area * rhoFace;

                    // No direct phi term for wall since phi_ghost = phi_int → delta_phi = 0
                    // No off-diagonal since ghost is not a real DOF

                    // === DENSITY RETARDATION FOR WALL FACES ===
                    // Wall flux = ρ_face * (V_int · m_wall) * A
                    // In supersonic regions, ρ_face depends on upwind (interior) density:
                    //   ρ_face = (1-μ) * ρ_isen + μ * ρ_upwind
                    // where upwind = interior cell for wall faces.
                    //
                    // ∂F/∂φ includes: (V_int · m_wall) * ∂ρ_face/∂φ_int * A
                    const uInt = this.spatialDisc_.uu_[elem];
                    const vInt = this.spatialDisc_.vv_[elem];
                    const vDotMwall = uInt * mWallX + vInt * mWallY;
                    
                    const mu_int = this.spatialDisc_.mumu_[elem];
                    
                    if mu_int > 0.0 {
                        const gamma = this.inputs_.GAMMA_;
                        const Minf2 = this.inputs_.MACH_ * this.inputs_.MACH_;
                        const rhoInt = this.spatialDisc_.rhorho_[elem];
                        const machInt = this.spatialDisc_.machmach_[elem];
                        const velMag2 = uInt*uInt + vInt*vInt;
                        const velMag = sqrt(velMag2);
                        
                        // ∂ρ_upwind/∂|V| = -ρ^(2-γ) * M∞² * |V|
                        const drho_dVelMag = -rhoInt**(2.0-gamma) * Minf2 * velMag;
                        
                        // ∂|V|/∂φ_int = -sumW · (V/|V|)
                        // For wall, gradient includes ghost contribution: use (-sumW + w_ghost)
                        var dVelMag_dPhi_int = 0.0;
                        if velMag > 1e-10 {
                            const dux_dPhi = -sumWx_elem + wx_elemToNeighbor;
                            const dvy_dPhi = -sumWy_elem + wy_elemToNeighbor;
                            dVelMag_dPhi_int = (dux_dPhi * uInt + dvy_dPhi * vInt) / velMag;
                        }
                        
                        // Get isentropic density at face
                        const rhoIsen = if mu_int < 0.999 
                                        then (rhoFace - mu_int * rhoInt) / (1.0 - mu_int)
                                        else rhoInt;
                        
                        // Term 1: μ * ∂ρ_upwind/∂φ_int
                        var drhoFace_dPhi_int = mu_int * drho_dVelMag * dVelMag_dPhi_int;
                        
                        // Term 2: (∂μ/∂φ_int) * (ρ_upwind - ρ_isen)
                        if velMag > 1e-10 {
                            const dmu_dVelMag = this.inputs_.MU_C_ * 2.0 * machInt * machInt / velMag;
                            const dmu_dPhi_int = dmu_dVelMag * dVelMag_dPhi_int;
                            drhoFace_dPhi_int += dmu_dPhi_int * (rhoInt - rhoIsen);
                        }
                        
                        // Add contribution: (V_int · m_wall) * ∂ρ_face/∂φ_int * A
                        const densityRetardation = vDotMwall * drhoFace_dPhi_int * area;
                        diag += sign * densityRetardation;
                    }

                    // === CIRCULATION (Γ) DERIVATIVE FOR WALL FACES ===
                    // Wall flux = V_int · m_wall * A, where V_int = ∇φ_int
                    // If the interior cell has wake-crossing faces, ∇φ_int depends on Γ
                    // ∂flux/∂Γ = (∂∇φ_int/∂Γ · m_wall) * A
                    const dgradX_elem = this.dgradX_dGamma_[elem];
                    const dgradY_elem = this.dgradY_dGamma_[elem];
                    const dFlux_dGamma_wall = (dgradX_elem * mWallX + dgradY_elem * mWallY) * sign * area * rhoFace;
                    dRes_dGamma += dFlux_dGamma_wall;

                } else {
                    // === FARFIELD BOUNDARY FACE ===
                    // Farfield BC: phi_ghost = U_inf*x + V_inf*y (Dirichlet, fixed)
                    //              V_ghost = 2*V_inf - V_int
                    //
                    // This gives: V_avg = V_inf (constant, no phi dependency)
                    //
                    // The interior gradient includes ghost as neighbor:
                    //   gradPhi_int includes w_int_to_ghost * (phi_ghost - phi_int)
                    //   d(gradPhi_int)/d(phi_int) = -sumW_int + w_int_to_ghost * 0 = -sumW_int
                    //   (phi_ghost is fixed, so d(phi_ghost)/d(phi_int) = 0)
                    //
                    // However, the averaged velocity V_avg = V_inf is constant, so the flux
                    // at farfield faces doesn't depend on interior phi through the gradient.
                    // The only dependency is through the direct term.
                    
                    // Diagonal contribution from d(0.5*(gradPhi_elem · m))/d(phi_elem)
                    // V_avg = V_inf is constant, but we still have the correction term
                    // delta = V_avg · t - dPhi/dL, and dPhi = phi_ghost - phi_int
                    // where phi_ghost is fixed
                    
                    // Actually for farfield, V_avg = V_inf (constant), so V_avg · m is constant
                    // The only contribution is from the direct term: k*invL*(phi_ghost - phi_int)
                    // d/d(phi_int) = -k*invL = -directCoeff
                    
                    // Apply sign and area for direct term only
                    diag -= directCoeff * area * rhoFace;
                }
            }
            
            // Add diagonal entry
            this.A_petsc.add(elem-1, elem-1, diag * this.spatialDisc_.invElemVolume_[elem]);
            // Store diag for possible use in upwinding
            this.Jij_[elem] = diag;
            
            // Add dRes/dΓ entry (column for circulation)
            this.A_petsc.add(elem-1, this.gammaIndex_, dRes_dGamma * this.spatialDisc_.invElemVolume_[elem]);
        }

        // === BETA-BASED UPWIND AUGMENTATION (element-centric for parallelization) ===
        // Loop over elements instead of faces to avoid race conditions.
        // Each element checks its faces to see if it's the downwind cell of a supersonic face.
        // Reuses upwindElem_ computed during artificial density calculation.
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            const faces = this.spatialDisc_.mesh_.elem2edge_[
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 ..
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
                
            for face in faces {
                const machFace = this.spatialDisc_.machFace_[face];
                
                if machFace >= this.inputs_.MACH_C_ {
                    const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                    const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                    
                    // Both cells must be interior (not ghost cells)
                    if elem1 <= this.spatialDisc_.nelemDomain_ && elem2 <= this.spatialDisc_.nelemDomain_ {
                        // Reuse upwind/downwind elements computed during artificial density
                        const upwindElem = this.spatialDisc_.upwindElem_[face];
                        const downwindElem = this.spatialDisc_.downwindElem_[face];
                        
                        // Only process if this element is the downwind cell
                        // This ensures each matrix entry is only written by one task
                        if downwindElem == elem {
                            // Use precomputed invL_IJ_ (inverse of cell centroid distance)
                            const increase = this.inputs_.BETA_ * this.spatialDisc_.velMagFace_[face] 
                            * this.spatialDisc_.invL_IJ_[face] * this.spatialDisc_.invElemVolume_[elem];
                            
                            // Increase absolute value of diagonal term for downwind element
                            const diagTerm = this.Jij_[elem];
                            if diagTerm >= 0.0 {
                                // Increase diagonal and decrease off-diagonal
                                this.A_petsc.add(elem-1, elem-1, increase);
                                this.A_petsc.add(elem-1, upwindElem-1, -increase);
                            } else {
                                // Decrease diagonal and increase off-diagonal
                                this.A_petsc.add(elem-1, elem-1, -increase);
                                this.A_petsc.add(elem-1, upwindElem-1, increase);
                            }
                        }
                    }
                }
            }
        }
        
        // === KUTTA CONDITION ROW ===
        // Γ - (φ_upper,TE - φ_lower,TE) = 0
        // Currently, circulation is computed as:
        //   Γ = φ_upper + V_upper·Δs - [φ_lower + V_lower·Δs]
        // Simplified for linearization: Γ ≈ φ_upper - φ_lower (main cells)
        //
        // Kutta residual: R_Γ = Γ - (φ_upperTE - φ_lowerTE)
        // d(R_Γ)/dΓ = 1
        // d(R_Γ)/dφ_upperTE = -1
        // d(R_Γ)/dφ_lowerTE = +1
        
        // dKutta/dΓ = 1
        this.A_petsc.add(this.gammaIndex_, this.gammaIndex_, 1.0 / (this.spatialDisc_.elemVolume_[this.spatialDisc_.upperTEelem_] + this.spatialDisc_.elemVolume_[this.spatialDisc_.lowerTEelem_]));
        
        // dKutta/dφ_upperTE1 = -1.0
        this.A_petsc.add(this.gammaIndex_, this.spatialDisc_.upperTEelem_ - 1, -1.0 / (this.spatialDisc_.elemVolume_[this.spatialDisc_.upperTEelem_] + this.spatialDisc_.elemVolume_[this.spatialDisc_.lowerTEelem_]));
        // dKutta/dφ_lowerTE1 = +1.0
        this.A_petsc.add(this.gammaIndex_, this.spatialDisc_.lowerTEelem_ - 1, 1.0 / (this.spatialDisc_.elemVolume_[this.spatialDisc_.upperTEelem_] + this.spatialDisc_.elemVolume_[this.spatialDisc_.lowerTEelem_]));
        
        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
        
        // Extract CSR matrix for native GMRES if enabled
        if this.inputs_.USE_NATIVE_GMRES_ {
            this.extractCSRFromPETSc();
            this.nativeGmres_.setupPreconditioner(this.A_csr_);
        }
    }
    
    // Build CSR sparsity pattern from mesh connectivity (call once during init)
    proc buildCSRSparsityPattern() {
        const n = this.spatialDisc_.nelemDomain_ + 1;
        
        // Count non-zeros per row based on mesh connectivity
        var nnzPerRow: [0..#n] int;
        
        for elem in 1..this.spatialDisc_.nelemDomain_ {
            // Diagonal
            var count = 1;
            // Neighbors
            const faces = this.spatialDisc_.mesh_.elem2edge_[
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. 
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
            for face in faces {
                const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;
                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    count += 1;
                }
            }
            // Connection to circulation
            count += 1;
            nnzPerRow[elem-1] = count;
        }
        // Last row (Kutta condition): connected to Γ, upper TE, lower TE
        nnzPerRow[n-1] = 5;
        
        // Total non-zeros
        var totalNnz = 0;
        for i in 0..#n do totalNnz += nnzPerRow[i];
        
        // Initialize CSR structure
        this.A_csr_ = new SparseMatrixCSR(n, totalNnz);
        
        // Build row pointers
        var k = 0;
        for i in 0..#n {
            this.A_csr_.rowPtr[i] = k;
            k += nnzPerRow[i];
        }
        this.A_csr_.rowPtr[n] = k;
        
        // Build column indices (sorted order within each row)
        for elem in 1..this.spatialDisc_.nelemDomain_ {
            const row = elem - 1;
            var cols: [0..#nnzPerRow[row]] int;
            var idx = 0;
            
            // Collect all column indices
            cols[idx] = row;  // diagonal
            idx += 1;
            
            const faces = this.spatialDisc_.mesh_.elem2edge_[
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. 
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
            for face in faces {
                const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;
                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    cols[idx] = neighbor - 1;
                    idx += 1;
                }
            }
            // Connection to circulation (last column)
            cols[idx] = n - 1;
            idx += 1;
            
            // Sort columns
            sort(cols);
            
            // Copy to CSR
            const rowStart = this.A_csr_.rowPtr[row];
            for i in 0..#idx {
                this.A_csr_.colIdx[rowStart + i] = cols[i];
            }
        }
        
        // Last row (Kutta): Γ, upper TE, lower TE
        const gammaRow = n - 1;
        const rowStart = this.A_csr_.rowPtr[gammaRow];
        var kuttaCols: [0..#5] int;
        kuttaCols[0] = this.spatialDisc_.lowerTEelem_ - 1;
        kuttaCols[1] = this.spatialDisc_.upperTEelem_ - 1;
        kuttaCols[2] = gammaRow;  // diagonal
        // Note: actual nnz might be less, just use what's needed
        sort(kuttaCols[0..2]);
        for i in 0..2 {
            this.A_csr_.colIdx[rowStart + i] = kuttaCols[i];
        }
        // Pad remaining
        for i in 3..4 {
            this.A_csr_.colIdx[rowStart + i] = gammaRow;
        }
        
        this.A_csr_.buildDiagonalIndex();
    }
    
    // Update CSR values from PETSc matrix (called each iteration)
    proc extractCSRFromPETSc() {
        const n = this.spatialDisc_.nelemDomain_ + 1;
        
        // Copy values using known sparsity pattern
        for i in 0..#n {
            const rowStart = this.A_csr_.rowPtr[i];
            const rowEnd = this.A_csr_.rowPtr[i + 1];
            for k in rowStart..rowEnd-1 {
                const j = this.A_csr_.colIdx[k];
                this.A_csr_.values[k] = this.A_petsc.get(i, j);
            }
        }
    }

    proc initialize() {
        this.spatialDisc_.initializeMetrics();
        this.spatialDisc_.initializeKuttaCells();
        this.spatialDisc_.initializeSolution();
        this.spatialDisc_.run();
        this.initializeJacobian();
        this.computeGradientSensitivity();
        
        // Build CSR sparsity pattern for native GMRES (must be after initializeJacobian)
        if this.inputs_.USE_NATIVE_GMRES_ {
            this.buildCSRSparsityPattern();
        }
        
        this.computeJacobian();

        // Initialize SFD filtered state to initial solution
        if this.inputs_.SFD_ENABLED_ {
            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.phi_bar_[elem] = this.spatialDisc_.phi_[elem];
            }
            this.circulation_bar_ = this.spatialDisc_.circulation_;
        }

        if this.inputs_.START_FILENAME_ != "" {
            writeln("Initializing solution from file: ", this.inputs_.START_FILENAME_);
            const (xElem, yElem, rho, phi, it, time, res, cl, cd, cm, circulation) = readSolution(this.inputs_.START_FILENAME_);
            for i in it.domain {
                this.timeList_.pushBack(time[i]);
                this.itList_.pushBack(it[i]);
                this.resList_.pushBack(res[i]);
                this.clList_.pushBack(cl[i]);
                this.cdList_.pushBack(cd[i]);
                this.cmList_.pushBack(cm[i]);
                this.circulationList_.pushBack(circulation[i]);
            }
            this.it_ = it.last;
            this.t0_ = time.last;
            this.first_res_ = res.first;
        }
    }

    // Reset solver state for Mach continuation (keeps current solution as initial guess)
    // newInputs should have the updated Mach number and derived quantities
    proc resetForContinuation(ref newInputs: potentialInputs) {
        this.it_ = 0;
        
        // Update the inputs record in this class and in spatialDiscretization
        this.inputs_ = newInputs;
        this.spatialDisc_.updateInputs(newInputs);
        
        // Recompute Jacobian sparsity pattern (in case it changed, though typically it doesn't)
        this.computeGradientSensitivity();
        this.computeJacobian();

        // Reset SFD filtered state to current solution
        if this.inputs_.SFD_ENABLED_ {
            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.phi_bar_[elem] = this.spatialDisc_.phi_[elem];
            }
            this.circulation_bar_ = this.spatialDisc_.circulation_;
        }
    }

    proc solve() {
        var normalized_res: real(64) = 1e12;
        var res : real(64) = 1e12;        // Current residual (absolute)
        var res_prev : real(64) = 1e12;  // Previous iteration residual for line search
        var omega : real(64) = this.inputs_.OMEGA_;  // Current relaxation factor
        const OMEGA_MIN : real(64) = 0.1;  // Minimum allowed OMEGA
        const OMEGA_DECREASE : real(64) = 0.5;  // Factor to decrease OMEGA in line search
        var time: stopwatch;
        
        // Arrays to store state for line search (backtracking)
        var phi_backup: [1..this.spatialDisc_.nelemDomain_] real(64);
        var circulation_backup: real(64);
        
        // // Compute initial residual to check if already converged (useful for Mach continuation)
        // this.spatialDisc_.run();
        // res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
        // this.first_res_ = res;
        // normalized_res = 1.0;
        // Convergence check: either relative tolerance OR absolute tolerance
        // The absolute tolerance is important for Mach continuation where the 
        // initial residual may already be very small
        while ((normalized_res > this.inputs_.CONV_TOL_ && res > this.inputs_.CONV_ATOL_) && this.it_ < this.inputs_.IT_MAX_ && isNan(normalized_res) == false) {
            this.it_ += 1;
            time.start();

            this.computeJacobian();
            
            // Always use full Newton step in RHS (omega will be applied to the solution update)
            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.b_petsc.set(elem-1, -this.spatialDisc_.res_[elem]);
            }
            this.b_petsc.set(this.gammaIndex_, -this.spatialDisc_.kutta_res_);
            this.b_petsc.assemblyComplete();

            var its: int;
            var reason: int;
            
            // === ADAPTIVE (INEXACT) NEWTON: Eisenstat-Walker forcing terms ===
            // Idea: Use loose inner tolerance early (save GMRES iterations), 
            //       tighten as outer residual decreases for quadratic convergence
            // Reference: Eisenstat & Walker, SIAM J. Sci. Comput. 17(1), 1996
            var current_rtol = this.inputs_.GMRES_RTOL_;
            if this.inputs_.ADAPTIVE_GMRES_TOL_ && normalized_res < 1e12 {
                // Eisenstat-Walker Choice 1: eta_k = |r_k - r_{k-1}| / |r_{k-1}|
                // Simplified version: eta = eta_max * (res / first_res)^gamma
                // This gives loose tolerance early and tight tolerance near convergence
                const eta = this.inputs_.GMRES_RTOL_ETA_;
                current_rtol = max(this.inputs_.GMRES_RTOL_MIN_,
                                   min(this.inputs_.GMRES_RTOL_MAX_, 
                                       eta * normalized_res));
                // Update tolerances in solvers
                if this.inputs_.USE_NATIVE_GMRES_ {
                    this.nativeGmres_.setTolerance(current_rtol, this.inputs_.GMRES_ATOL_);
                } else {
                    this.ksp.setTolerances(current_rtol, this.inputs_.GMRES_ATOL_, 
                                          this.inputs_.GMRES_DTOL_, this.inputs_.GMRES_MAXIT_);
                }
            }

            if this.inputs_.USE_NATIVE_GMRES_ {
                // === NATIVE CHAPEL GMRES ===
                const nDOF = this.spatialDisc_.nelemDomain_ + 1;
                
                // Copy RHS to native arrays
                forall i in 0..#nDOF {
                    this.b_native_[i] = this.b_petsc.get(i);
                    this.x_native_[i] = 0.0;  // Zero initial guess
                }
                
                if this.inputs_.JACOBIAN_TYPE_ == "numerical" && this.jvpProvider_ != nil {
                    // === MATRIX-FREE NUMERICAL JACOBIAN ===
                    // Store current residual R(phi) for finite difference
                    var R0_native: [0..#this.spatialDisc_.nelemDomain_] real(64);
                    forall i in 0..#this.spatialDisc_.nelemDomain_ {
                        R0_native[i] = this.spatialDisc_.res_[i+1];
                    }
                    
                    // Borrow the JVP provider
                    var jvp: borrowed JacobianVectorProductProvider = this.jvpProvider_!.borrow();
                    
                    // Set state in JVP provider
                    jvp.setState(
                        this.spatialDisc_.phi_, 
                        this.spatialDisc_.circulation_,
                        R0_native, 
                        this.spatialDisc_.kutta_res_
                    );
                    
                    // Always extract CSR and update preconditioner
                    this.extractCSRFromPETSc();
                    
                    const (nativeIts, nativeReason) = this.nativeGmres_.solveMatrixFreeRight(
                        this.A_csr_, this.x_native_, this.b_native_,
                        jvp, true
                    );
                    its = nativeIts;
                    reason = nativeReason : int;
                } else {
                    // === ANALYTICAL JACOBIAN (standard matrix-based) ===
                    this.extractCSRFromPETSc();
                    const (nativeIts, nativeReason) = this.nativeGmres_.solve(this.A_csr_, this.x_native_, this.b_native_, true);
                    its = nativeIts;
                    reason = nativeReason : int;
                }
                
                // Copy solution back to PETSc vector for line search compatibility
                forall i in 0..#nDOF {
                    this.x_petsc.set(i, this.x_native_[i]);
                }
                this.x_petsc.assemblyComplete();
            } else {
                // === PETSC GMRES ===
                const (petscIts, petscReason) = GMRES(this.ksp, this.A_petsc, this.b_petsc, this.x_petsc);
                its = petscIts;
                reason = petscReason;
            }
            
            var lineSearchIts = 0;
            omega = this.inputs_.OMEGA_;
            
            if this.inputs_.LINE_SEARCH_ {
                // === BACKTRACKING LINE SEARCH ===
                // Save current state
                forall elem in 1..this.spatialDisc_.nelemDomain_ {
                    phi_backup[elem] = this.spatialDisc_.phi_[elem];
                }
                circulation_backup = this.spatialDisc_.circulation_;
                
                const MAX_LINE_SEARCH = this.inputs_.MAX_LINE_SEARCH_;
                const SUFFICIENT_DECREASE = this.inputs_.SUFFICIENT_DECREASE_;  // Allow up to 20% increase (inexact Newton)
                var accepted = false;
                
                while !accepted && lineSearchIts < MAX_LINE_SEARCH {
                    // Apply update with current omega
                    forall elem in 1..this.spatialDisc_.nelemDomain_ {
                        this.spatialDisc_.phi_[elem] = phi_backup[elem] + omega * this.x_petsc.get(elem-1);
                    }
                    this.spatialDisc_.circulation_ = circulation_backup + omega * this.x_petsc.get(this.gammaIndex_);
                    
                    // Compute new residual
                    this.spatialDisc_.run();
                    res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
                    
                    // Accept if residual doesn't increase too much (allows inexact Newton behavior)
                    if res < res_prev * SUFFICIENT_DECREASE || this.it_ == 1 {
                        accepted = true;
                    } else {
                        // Reduce omega and retry
                        omega *= OMEGA_DECREASE;
                        if omega < OMEGA_MIN {
                            omega = OMEGA_MIN;
                            accepted = true;  // Accept anyway with minimum omega
                        }
                        lineSearchIts += 1;
                    }
                }
            } else {
                // === NO LINE SEARCH - fixed omega ===
                forall elem in 1..this.spatialDisc_.nelemDomain_ {
                    this.spatialDisc_.phi_[elem] += omega * this.x_petsc.get(elem-1);
                }
                this.spatialDisc_.circulation_ += omega * this.x_petsc.get(this.gammaIndex_);
                
                // Compute residual for convergence check
                this.spatialDisc_.run();
                res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
            }

            // === SELECTIVE FREQUENCY DAMPING (SFD) ===
            if this.inputs_.SFD_ENABLED_ {
                const chi = this.inputs_.SFD_CHI_;
                const delta = this.inputs_.SFD_DELTA_;
                const dt = 1.0;  // One Newton iteration = one time unit
                
                // Compute the exponential factor E = exp(-(χ + 1/Δ)·Δt)
                const exponent = -(chi + 1.0/delta) * dt;
                const E = exp(exponent);
                
                // Matrix exponential coefficients from Jordi Eq. (9)
                const scale = 1.0 / (1.0 + chi * delta);
                const M11 = scale * (1.0 + chi * delta * E);        // coefficient for q from q*
                const M12 = scale * chi * delta * (1.0 - E);        // coefficient for q from q̄
                const M21 = scale * (1.0 - E);                      // coefficient for q̄ from q*
                const M22 = scale * (chi * delta + E);              // coefficient for q̄ from q̄
                
                // Apply matrix exponential to (φ*, φ̄) where φ* = Φ(φ^n) is Newton result
                forall elem in 1..this.spatialDisc_.nelemDomain_ {
                    const phi_star = this.spatialDisc_.phi_[elem];  // Result from Newton (Φ(q^n))
                    const phi_bar = this.phi_bar_[elem];            // Previous filtered state
                    
                    // Apply encapsulated SFD transform
                    const phi_new = M11 * phi_star + M12 * phi_bar;
                    const phi_bar_new = M21 * phi_star + M22 * phi_bar;
                    
                    this.spatialDisc_.phi_[elem] = phi_new;
                    this.phi_bar_[elem] = phi_bar_new;
                }
                
                // Apply same transform to circulation
                const gamma_star = this.spatialDisc_.circulation_;
                const gamma_bar = this.circulation_bar_;
                this.spatialDisc_.circulation_ = M11 * gamma_star + M12 * gamma_bar;
                this.circulation_bar_ = M21 * gamma_star + M22 * gamma_bar;

                // Recompute residual after SFD modification
                this.spatialDisc_.run();
                res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
            }
            
            res_prev = res;
            
            if this.it_ == 1 {
                this.first_res_ = res;
            }
            normalized_res = res / this.first_res_;

            const res_wall = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wall_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wall_dom]);
            const res_fluid = RMSE(this.spatialDisc_.res_[this.spatialDisc_.fluid_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.fluid_dom]);
            const res_wake = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wake_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wake_dom]);

            const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
            time.stop();
            const elapsed = time.elapsed() + this.t0_;
            if this.inputs_.ADAPTIVE_GMRES_TOL_ {
                writeln(" Time: ", elapsed, " It: ", this.it_,
                        " res: ", res, " norm res: ", normalized_res, " kutta res: ", this.spatialDisc_.kutta_res_,
                        " res wall: ", res_wall, " res fluid: ", res_fluid, " res wake: ", res_wake,
                        " Cl: ", Cl, " Cd: ", Cd, " Cm: ", Cm, " Circulation: ", this.spatialDisc_.circulation_,
                        " GMRES its: ", its, " reason: ", reason, " omega: ", omega, " LS its: ", lineSearchIts,
                        " GMRES rtol: ", current_rtol);
            } else {
                writeln(" Time: ", elapsed, " It: ", this.it_,
                        " res: ", res, " norm res: ", normalized_res, " kutta res: ", this.spatialDisc_.kutta_res_,
                        " res wall: ", res_wall, " res fluid: ", res_fluid, " res wake: ", res_wake,
                        " Cl: ", Cl, " Cd: ", Cd, " Cm: ", Cm, " Circulation: ", this.spatialDisc_.circulation_,
                        " GMRES its: ", its, " reason: ", reason, " omega: ", omega, " LS its: ", lineSearchIts);
            }

            this.timeList_.pushBack(elapsed);
            this.itList_.pushBack(this.it_);
            this.resList_.pushBack(res);
            this.clList_.pushBack(Cl);
            this.cdList_.pushBack(Cd);
            this.cmList_.pushBack(Cm);
            this.circulationList_.pushBack(this.spatialDisc_.circulation_);
            
            if this.it_ % this.inputs_.CGNS_OUTPUT_FREQ_ == 0 {
                this.spatialDisc_.writeSolution(this.timeList_, this.itList_, this.resList_, this.clList_, this.cdList_, this.cmList_, this.circulationList_);
            }
        }

        this.spatialDisc_.writeSolution(this.timeList_, this.itList_, this.resList_, this.clList_, this.cdList_, this.cmList_, this.circulationList_);

    }
}

}
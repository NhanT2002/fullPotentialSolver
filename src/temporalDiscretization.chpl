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

class temporalDiscretization {
    var spatialDisc_: shared spatialDiscretization;
    var inputs_: potentialInputs;
    var it_: int = 0;
    
    // Index for circulation DOF (last row/column) - must be before A_petsc for init order
    var gammaIndex_: int;
    
    var A_petsc : owned PETSCmatrix_c;
    var x_petsc : owned PETSCvector_c;
    var b_petsc : owned PETSCvector_c;
    var ksp : owned PETSCksp_c;

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

        // Initialize gradient sensitivity arrays
        this.gradSensitivity_dom = {1..spatialDisc.nelemDomain_};
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

        // // Debug: count cells with non-zero gradient sensitivity
        // if this.it_ == 1 {
        //     var count = 0;
        //     var maxDgx = 0.0, maxDgy = 0.0;
        //     for elem in 1..this.spatialDisc_.nelemDomain_ {
        //         if this.dgradX_dGamma_[elem] != 0.0 || this.dgradY_dGamma_[elem] != 0.0 {
        //             count += 1;
        //             if abs(this.dgradX_dGamma_[elem]) > maxDgx then maxDgx = abs(this.dgradX_dGamma_[elem]);
        //             if abs(this.dgradY_dGamma_[elem]) > maxDgy then maxDgy = abs(this.dgradY_dGamma_[elem]);
        //         }
        //     }
        //     writeln("DEBUG: Cells with non-zero grad sensitivity: ", count,
        //             " max|dgradX|: ", maxDgx, " max|dgradY|: ", maxDgy);
        // }
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
                    
                    this.A_petsc.add(elem-1, neighbor-1, offdiag);

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
            this.A_petsc.add(elem-1, elem-1, diag);
            // Store diag for possible use in upwinding
            this.Jij_[elem] = diag;
            
            // Add dRes/dΓ entry (column for circulation)
            this.A_petsc.add(elem-1, this.gammaIndex_, dRes_dGamma);
        }

        for face in 1..this.spatialDisc_.nface_ {
            const uFace = this.spatialDisc_.uFace_[face];
            const vFace = this.spatialDisc_.vFace_[face];
            const rhoFace = this.spatialDisc_.rhoFace_[face];
            const machFace = this.spatialDisc_.mach(uFace, vFace, rhoFace);
            if machFace >= this.inputs_.MACH_C_ {
                const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                if elem1 <= this.spatialDisc_.nelemDomain_ && elem2 <= this.spatialDisc_.nelemDomain_ {
                    // determine upwind and downwind elements
                    const nx = this.spatialDisc_.faceNormalX_[face];
                    const ny = this.spatialDisc_.faceNormalY_[face];
                    const velDotN = uFace * nx + vFace * ny;
                    var upwindElem: int;
                    var downwindElem: int;
                    if velDotN >= 0.0 {
                        upwindElem = elem1;
                        downwindElem = elem2;
                    } else {
                        upwindElem = elem2;
                        downwindElem = elem1;
                    }
                    // Increase absolute value of diagonal term for downwind element
                    const deltaS = sqrt((this.spatialDisc_.elemCentroidX_[downwindElem] - this.spatialDisc_.elemCentroidX_[upwindElem])**2 +
                                        (this.spatialDisc_.elemCentroidY_[downwindElem] - this.spatialDisc_.elemCentroidY_[upwindElem])**2);
                    const increase = this.inputs_.BETA_*sqrt(uFace*uFace + vFace*vFace) / deltaS;
                    // Extract main diagonal term for downwind element
                    const diagTerm = this.Jij_[downwindElem];
                    if diagTerm >= 0.0 {
                        // Increase diagonal and decrease off-diagonal
                        this.A_petsc.add(downwindElem-1, downwindElem-1, increase);
                        this.A_petsc.add(downwindElem-1, upwindElem-1, -increase);
                    }
                    else {
                        // Decrease diagonal and increase off-diagonal
                        this.A_petsc.add(downwindElem-1, downwindElem-1, -increase);
                        this.A_petsc.add(downwindElem-1, upwindElem-1, increase);
                    }
                }
            }
            
        }
        
        // === KUTTA CONDITION ROW ===
        // Γ - (φ_upper,TE - φ_lower,TE) = 0
        // Currently, circulation is computed as:
        //   Γ = 0.5*(φ_upper1 + φ_upper2) + V_face·Δs - [0.5*(φ_lower1 + φ_lower2) + V_face·Δs]
        // Simplified for linearization: Γ ≈ φ_upper1 - φ_lower1 (main cells)
        //
        // Kutta residual: R_Γ = Γ - (φ_upperTE - φ_lowerTE)
        // d(R_Γ)/dΓ = 1
        // d(R_Γ)/dφ_upperTE = -1
        // d(R_Γ)/dφ_lowerTE = +1
        
        // dKutta/dΓ = 1
        this.A_petsc.add(this.gammaIndex_, this.gammaIndex_, 1.0);
        
        // dKutta/dφ_upperTE1 = -1.0
        this.A_petsc.add(this.gammaIndex_, this.spatialDisc_.upperTEelem_ - 1, -1.0);
        // dKutta/dφ_lowerTE1 = +1.0
        this.A_petsc.add(this.gammaIndex_, this.spatialDisc_.lowerTEelem_ - 1, 1.0);
        
        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
    }

    proc initialize() {
        this.spatialDisc_.initializeMetrics();
        this.spatialDisc_.initializeKuttaCells();
        this.spatialDisc_.initializeSolution();
        this.initializeJacobian();
        this.computeGradientSensitivity();
        this.computeJacobian();
    }

    proc solve() {
        var normalized_res: real(64) = 1e12;
        var first_res : real(64) = 1e12;
        var time: stopwatch;
        while (normalized_res > this.inputs_.CONV_TOL_ && this.it_ < this.inputs_.IT_MAX_ && isNan(normalized_res) == false) {
            this.it_ += 1;
            time.start();

            this.spatialDisc_.run();

            this.computeJacobian();
            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.b_petsc.set(elem-1, -this.inputs_.OMEGA_ * this.spatialDisc_.res_[elem]);
            }
            this.b_petsc.set(this.gammaIndex_, -this.inputs_.OMEGA_ * this.spatialDisc_.kutta_res_);
            this.b_petsc.assemblyComplete();

            const (its, reason) = GMRES(this.ksp, this.A_petsc, this.b_petsc, this.x_petsc);

            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.spatialDisc_.phi_[elem] += this.x_petsc.get(elem-1);
            }
            
            // Update circulation from the Newton step
            this.spatialDisc_.circulation_ += this.x_petsc.get(this.gammaIndex_);

            const res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
            if this.it_ == 1 {
                first_res = res;
            }

            normalized_res = res / first_res;

            const res_wall = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wall_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wall_dom]);
            const res_fluid = RMSE(this.spatialDisc_.res_[this.spatialDisc_.fluid_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.fluid_dom]);
            const res_wake = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wake_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wake_dom]);

            const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
            time.stop();
            const elapsed = time.elapsed();
            writeln(" Time: ", elapsed, " It: ", this.it_,
                    " res: ", res, " norm res: ", normalized_res,
                    " res wall: ", res_wall, " res fluid: ", res_fluid, " res wake: ", res_wake,
                    " Cl: ", Cl, " Cd: ", Cd, " Cm: ", Cm, " Circulation: ", this.spatialDisc_.circulation_,
                    " GMRES its: ", its, " reason: ", reason);

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
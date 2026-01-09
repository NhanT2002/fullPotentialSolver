module unsteadyTemporalDiscretization
{
use mesh;
use writeCGNS;
use Math;
use linearAlgebra;
use Time;
import input.potentialInputs;
use unsteadySpatialDiscretization;
use PETSCapi;
use C_PETSC;
use petsc;
use CTypes;
use List;
use gmres;
use Sort;

class unsteadyTemporalDiscretization {
    var spatialDisc_: shared unsteadySpatialDiscretization;
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

    proc init(spatialDisc: shared unsteadySpatialDiscretization, ref inputs: potentialInputs) {
        writeln("Initializing temporal discretization...");
        this.spatialDisc_ = spatialDisc;
        this.inputs_ = inputs;

        const nWakeFace = this.spatialDisc_.wakeFace_.size;
        const M = spatialDisc.nelemDomain_*2 + nWakeFace;
        const N = spatialDisc.nelemDomain_*2 + nWakeFace;
        this.gammaIndex_ = spatialDisc.nelemDomain_*2;  // 0-based index for Γ
        
        this.A_petsc = new owned PETSCmatrix_c(PETSC_COMM_SELF, "seqaij", M, M, N, N);
        this.x_petsc = new owned PETSCvector_c(PETSC_COMM_SELF, N, N, 0.0, "seq");
        this.b_petsc = new owned PETSCvector_c(PETSC_COMM_SELF, N, N, 0.0, "seq");

        var nnz : [0..M-1] PetscInt;
        nnz = 10*(this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[1] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[1 + 1]].size + 1);
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

                        // Sign: +1 if elem above, neighbor below (φ_neighbor - Γ)
                        //       -1 if elem below, neighbor above (φ_neighbor + Γ)
                        const gammaSgn = if kuttaType_elem == 1 then -1.0 else 1.0;

                        dgx += wx * gammaSgn;
                        dgy += wy * gammaSgn;
                    }
                }

                this.dgradX_dGamma_[elem] = dgx;
                this.dgradY_dGamma_[elem] = dgy;
            }
        }
    }

    proc initializeJacobian() {
        // state vector x = [rho_1, rho2, ..., rho_N, phi_1, phi_2, ..., phi_N, Γ_1, ..., Γ_nWakeFaces]
        // Jacobian has block structure:
        // [ dRes^(rho)/d(rho)   dRes^(rho)/d(phi)   dRes^(rho)/d(Γ) ]
        // [ dRes^(phi)/d(rho)   dRes^(phi)/d(phi)   dRes^(phi)/d(Γ) ]
        // [       0             dRes^(wake)/d(phi)  dRes^(wake)/d(Γ) ]
        
        // first block: dRes^(rho)/d(rho)
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
        }
        // second block: dRes^(rho)/d(phi)
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            this.A_petsc.set(elem-1, this.spatialDisc_.nelemDomain_ + elem-1, 0.0);
            const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
            for face in faces {
                const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;
                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    this.A_petsc.set(elem-1, this.spatialDisc_.nelemDomain_ + neighbor-1, 0.0);
                }
            }
        }
        // third block: dRes^(rho)/d(Γ)
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
            for elem in [elem1, elem2] {
                this.A_petsc.set(elem-1, 2*this.spatialDisc_.nelemDomain_ + i -1, 0.0);
            }
        }
        // fourth block: dRes^(phi)/d(rho)
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, elem-1, 0.0);
        }
        // fifth block: dRes^(phi)/d(phi)
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, this.spatialDisc_.nelemDomain_ + elem-1, 0.0);
            const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
            for face in faces {
                const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;
                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, this.spatialDisc_.nelemDomain_ + neighbor-1, 0.0);
                }
            }
        }
        // sixth block: dRes^(phi)/d(Γ)
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];

            for elem in [elem1, elem2] {
                this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, 2*this.spatialDisc_.nelemDomain_ + i -1, 0.0);
            }
        }

        // seventh block: dRes^(wake)/d(phi)
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
            for elem in [elem1, elem2] {
                this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, this.spatialDisc_.nelemDomain_ + elem -1, 0.0);

                const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
                for face in faces {
                    const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                    const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                    const neighbor = if elem1 == elem then elem2 else elem1;
                    if neighbor <= this.spatialDisc_.nelemDomain_ {
                        this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, this.spatialDisc_.nelemDomain_ + neighbor-1, 0.0);
                    }
                }

            }
        }
        // eighth block: dRes^(wake)/d(Γ)
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
            for elem in [elem1, elem2] {
                this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, 2*this.spatialDisc_.nelemDomain_ + i -1, 0.0);
            }
        }

        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
    }

    proc computeJacobian() {
        this.A_petsc.zeroEntries();

        this.assemble_dResRho_dRho();
        this.assemble_dResRho_dPhi();
        this.assemble_dResRho_dGamma();
        this.assemble_dResPhi_dRho();
        this.assemble_dResPhi_dPhi();
        this.assemble_dResPhi_dGamma();
        this.assemble_dResWake_dPhi();
        this.assemble_dResWake_dGamma();
        
        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
    }

    proc assemble_dResRho_dRho() {
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            // Res = (V/dt) * (rho_elem - rho_elem^n) + sum_faces ( rho_f * (V·n) * area_face )
            // rho_f = 0.5 * (rho_elem + rho_neighbor)
            var diag = this.spatialDisc_.elemVolume_[elem] / this.inputs_.TIME_STEP_;
            const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
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

                // Average velocity at face
                const uFace = this.spatialDisc_.uFace_[face];
                const vFace = this.spatialDisc_.vFace_[face];

                diag += 0.5 * sign * (uFace * nx + vFace * ny) * area;

                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    const offDiag = 0.5 * sign * (uFace * nx + vFace * ny) * area;
                    this.A_petsc.set(elem-1, neighbor-1, offDiag);
                }
            }
            this.A_petsc.set(elem-1, elem-1, diag);
        }
    }

    proc assemble_dResRho_dPhi() {
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            var diag = 0.0;
            const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
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
                    const gradContrib = sign * face_diag * area * rhoFace;
                    diag += gradContrib;
                    
                    // Direct phi term: d((phi_2 - phi_1) * directCoeff)/d(phi_elem)
                    // For elem = elem1: d(phi2-phi1)/dphi1 = -1, sign=+1 → -directCoeff
                    // For elem = elem2: d(phi2-phi1)/dphi2 = +1, sign=-1 → -directCoeff
                    // Combined: always -directCoeff * ρ * A (no sign multiplication)
                    const directContrib = -directCoeff * area * rhoFace;
                    diag += directContrib;
                    
                    // === OFF-DIAGONAL CONTRIBUTION ===
                    const sumWx_neighbor = this.spatialDisc_.lsGradQR_!.sumWx_[neighbor];
                    const sumWy_neighbor = this.spatialDisc_.lsGradQR_!.sumWy_[neighbor];
                    
                    // From d(0.5*(gradPhi_elem · m))/d(phi_neighbor) = 0.5 * (w_elemToNeighbor · m)
                    var offdiag = 0.5 * (wx_elemToNeighbor * mx + wy_elemToNeighbor * my);
                    
                    // From d(0.5*(gradPhi_neighbor · m))/d(phi_neighbor) = 0.5 * (-sumW_neighbor · m)
                    offdiag += 0.5 * (-sumWx_neighbor * mx - sumWy_neighbor * my);
                    
                    // Apply sign and area to gradient terms
                    offdiag *= sign * area * rhoFace;
                    
                    // Direct phi term: d((phi_2 - phi_1) * directCoeff)/d(phi_neighbor)
                    // For elem1's residual, neighbor=elem2, d(phi2-phi1)/dphi2 = +1
                    // For elem2's residual, neighbor=elem1, d(phi2-phi1)/dphi1 = -1
                    // With sign factor: sign * d(phi2-phi1)/dphi_neighbor
                    //   elem is elem1: sign=+1, neighbor=elem2: d/dphi2 = +1 → contribution = +directCoeff
                    //   elem is elem2: sign=-1, neighbor=elem1: d/dphi1 = -1 → contribution = +directCoeff
                    // So regardless of which side elem is on, the direct contribution is +directCoeff
                    // BUT wait - this doesn't match the diagonal analysis. Let me reconsider...
                    //
                    // Actually, for elem being elem1:
                    //   R_elem1 = +1 * flux = ρ*(V·m)*A
                    //   V·m includes directCoeff*(phi2-phi1)
                    //   dR_elem1/dphi2 = +1 * ρ * A * (+directCoeff) = +ρ*A*directCoeff
                    //
                    // For elem being elem2:
                    //   R_elem2 = -1 * flux = -ρ*(V·m)*A  
                    //   dR_elem2/dphi1 = -1 * ρ * A * (-directCoeff) = +ρ*A*directCoeff
                    //
                    // So the direct term contributes +directCoeff*ρ*A to off-diagonal ALWAYS (no sign)
                    offdiag += directCoeff * area * rhoFace;
                    this.A_petsc.set(elem-1, this.spatialDisc_.nelemDomain_ + neighbor-1, offdiag);

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
            this.A_petsc.set(elem-1, this.spatialDisc_.nelemDomain_ + elem-1, diag);
        }
    }

    proc assemble_dResRho_dGamma() {
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];

            for elem in [elem1, elem2] {
                var dRes_dGamma = 0.0;  // Contribution d(res_elem)/d(Γ) for wake-crossing cells
                const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
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

                    const kuttaType_elem = this.spatialDisc_.kuttaCell_[elem];
                    const kuttaType_neighbor = this.spatialDisc_.kuttaCell_[neighbor];
                    var gammaSgn = 0.0;
                    if kuttaType_elem == 1 && kuttaType_neighbor == -1 {
                        gammaSgn = -1.0;  // elem above, neighbor below: +Γ
                    } else {
                        gammaSgn = 1.0; // elem below, neighbor above: -Γ
                    }
                    dFlux_dGamma += directCoeff * area * rhoFace * gammaSgn;

                    dRes_dGamma += dFlux_dGamma;
                }

                this.A_petsc.set(elem-1, 2*this.spatialDisc_.nelemDomain_ + i -1, dRes_dGamma);
            }
        }
    }

    proc assemble_dResPhi_dRho() {
        // Res = V * [(φ^n - φ^{n-1}) / dt + 0.5*(u^2 + v^2 - 1) + (rho^{gamma-1} - 1)/((gamma -1) * M_inf^2)]
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            const diag = this.spatialDisc_.elemVolume_[elem] * (this.spatialDisc_.rhorho_[elem]**(this.spatialDisc_.inputs_.GAMMA_ - 2)) / (this.inputs_.MACH_**2);
            this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, elem-1, diag);
        }
    }

    proc assemble_dResPhi_dPhi() {
        // Res = V * [(φ^n - φ^{n-1}) / dt + 0.5*(u^2 + v^2 - 1) + (rho^{gamma-1} - 1)/((gamma -1) * M_inf^2)]
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            var diag = 1.0 / this.inputs_.TIME_STEP_;

            const sumWx_elem = this.spatialDisc_.lsGradQR_!.sumWx_[elem];
            const sumWy_elem = this.spatialDisc_.lsGradQR_!.sumWy_[elem];
            diag += this.spatialDisc_.uu_[elem] * (-sumWx_elem) + this.spatialDisc_.vv_[elem] * (-sumWy_elem);
            
            const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
            for face in faces {
                const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    var wx_elemToNeighbor: real(64);
                    var wy_elemToNeighbor: real(64);
                    if elem1 == elem {
                        // elem is elem1, using weights from perspective 1
                        wx_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wxFinal1_[face];
                        wy_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wyFinal1_[face];
                    } else {
                        // elem is elem2, using weights from perspective 2
                        wx_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wxFinal2_[face];
                        wy_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wyFinal2_[face];
                    }
                    const offDiag = this.spatialDisc_.uu_[elem] * wx_elemToNeighbor + this.spatialDisc_.vv_[elem] * wy_elemToNeighbor;
                    this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, this.spatialDisc_.nelemDomain_ + neighbor-1, this.spatialDisc_.elemVolume_[elem] * offDiag);
                }
            }

            this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, this.spatialDisc_.nelemDomain_ + elem-1, this.spatialDisc_.elemVolume_[elem] * diag);
        }
    }

    proc assemble_dResPhi_dGamma() {
        // Res = V * [(φ^n - φ^{n-1}) / dt + 0.5*(u^2 + v^2 - 1) + (rho^{gamma-1} - 1)/((gamma -1) * M_inf^2)]
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];

            for elem in [elem1, elem2] {
                const dgradX_elem = this.dgradX_dGamma_[elem];
                const dgradY_elem = this.dgradY_dGamma_[elem];
                const diag = this.spatialDisc_.elemVolume_[elem] * (this.spatialDisc_.uu_[elem] * dgradX_elem + this.spatialDisc_.vv_[elem] * dgradY_elem);
                this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, 2*this.spatialDisc_.nelemDomain_ + i -1, diag);
            }
        }
    }

    proc assemble_dResWake_dPhi() {
        // Res_wake = (Γ_i - Γ_i^{n-1}) / dt + 0.5 * [u_upper^2 + v_upper^2 - (u_lower^2 + v_lower^2)]
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
            for elem in [elem1, elem2] {
                const sign = if this.spatialDisc_.kuttaCell_[elem] == 1 then 1.0 else -1.0;
                const sumWx_elem = this.spatialDisc_.lsGradQR_!.sumWx_[elem];
                const sumWy_elem = this.spatialDisc_.lsGradQR_!.sumWy_[elem];
                const diag = sign * (this.spatialDisc_.uu_[elem] * (-sumWx_elem) + this.spatialDisc_.vv_[elem] * (-sumWy_elem));
                this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, this.spatialDisc_.nelemDomain_ + elem -1, diag);

                const faces = this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
                for face in faces {
                    const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
                    const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
                    const neighbor = if elem1 == elem then elem2 else elem1;

                    if neighbor <= this.spatialDisc_.nelemDomain_ {
                        var wx_elemToNeighbor: real(64);
                        var wy_elemToNeighbor: real(64);
                        if elem1 == elem {
                            // elem is elem1, using weights from perspective 1
                            wx_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wxFinal1_[face];
                            wy_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wyFinal1_[face];
                        } else {
                            // elem is elem2, using weights from perspective 2
                            wx_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wxFinal2_[face];
                            wy_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wyFinal2_[face];
                        }
                        const offDiag = sign * (this.spatialDisc_.uu_[elem] * wx_elemToNeighbor + this.spatialDisc_.vv_[elem] * wy_elemToNeighbor);
                        this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, this.spatialDisc_.nelemDomain_ + neighbor-1, offDiag);
                    }
                }

            }
        }
    }

    proc assemble_dResWake_dGamma() {
        // Res_wake = (Γ_i - Γ_i^{n-1}) / dt + 0.5 * [u_upper^2 + v_upper^2 - (u_lower^2 + v_lower^2)]
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
            for elem in [elem1, elem2] {
                var diag = 1.0 / this.inputs_.TIME_STEP_;
                const sign = if this.spatialDisc_.kuttaCell_[elem] == 1 then 1.0 else -1.0;
                diag += sign * (this.spatialDisc_.uu_[elem] * this.dgradX_dGamma_[elem] + this.spatialDisc_.vv_[elem] * this.dgradY_dGamma_[elem]);
                this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, 2*this.spatialDisc_.nelemDomain_ + i -1, diag);
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
        
        this.computeJacobian();

        if this.inputs_.START_FILENAME_ != "" {
            writeln("Initializing solution from file: ", this.inputs_.START_FILENAME_);
            const (xElem, yElem, rho, phi, it, time, res, cl, cd, cm, circulation, wakeGamma) = readSolution(this.inputs_.START_FILENAME_);
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

        // Reset adaptive upwinding state for new Mach number
        this.inputs_.upwindAdapted_ = false;
        if this.inputs_.ADAPTIVE_UPWIND_ {
            this.spatialDisc_.inputs_.MU_C_ = this.inputs_.MU_C_START_;
            this.spatialDisc_.inputs_.MACH_C_ = this.inputs_.MACH_C_START_;
            writeln("  >>> Adaptive upwinding: starting with MU_C=", this.inputs_.MU_C_, ", MACH_C=", this.inputs_.MACH_C_);
        }
        
        // Reset adaptive BETA state for new Mach number
        this.inputs_.betaAdapted_ = false;
        if this.inputs_.ADAPTIVE_BETA_ {
            this.inputs_.BETA_ = this.inputs_.BETA_START_;
            writeln("  >>> Adaptive BETA: starting with BETA=", this.inputs_.BETA_);
        }
    }

    proc solve() {
        var normalized_res: real(64) = 1e12;
        var res : real(64) = 1e12;        // Current residual (absolute)
        var resPhi : real(64);      // Residual for phi equation
        var resWake : real(64);     // Residual for wake equation
        var omega : real(64) = this.inputs_.OMEGA_;  // Current relaxation factor
        const OMEGA_MIN : real(64) = 0.1;  // Minimum allowed OMEGA
        const OMEGA_DECREASE : real(64) = 0.5;  // Factor to decrease OMEGA in line search
        var time: stopwatch;
        
        // Arrays to store state for line search (backtracking)
        var phi_backup: [1..this.spatialDisc_.nelemDomain_] real(64);
        var circulation_backup: real(64);

        // Initial residual
        res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
        resPhi = RMSE(this.spatialDisc_.resPhi_, this.spatialDisc_.elemVolume_);
        resWake = RMSE(this.spatialDisc_.resWake_);
        if this.inputs_.START_FILENAME_ == "" {
            this.first_res_ = res;
        }
        normalized_res = res / this.first_res_;
        const res_wall = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wall_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wall_dom]);
        const res_fluid = RMSE(this.spatialDisc_.res_[this.spatialDisc_.fluid_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.fluid_dom]);
        const res_wake = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wake_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wake_dom]);
        const res_shock = RMSE(this.spatialDisc_.res_[this.spatialDisc_.shock_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.shock_dom]);
        const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
        const elapsed = this.t0_;
        writeln(" Time: ", elapsed, " It: ", this.it_,
                " res: ", res, " norm res: ", normalized_res, " phi res: ", resPhi, " wake res: ", resWake,
                " res wall: ", res_wall, " res fluid: ", res_fluid, " res wake: ", res_wake, " res shock: ", res_shock,
                " Cl: ", Cl, " Cd: ", Cd, " Cm: ", Cm, " Circulation: ", this.spatialDisc_.circulation_);
        this.timeList_.pushBack(elapsed);
        this.itList_.pushBack(this.it_);
        this.resList_.pushBack(res);
        this.clList_.pushBack(Cl);
        this.cdList_.pushBack(Cd);
        this.cmList_.pushBack(Cm);
        this.circulationList_.pushBack(this.spatialDisc_.circulation_);

        while ((normalized_res > this.inputs_.CONV_TOL_ && res > this.inputs_.CONV_ATOL_) && this.it_ < this.inputs_.IT_MAX_ && isNan(normalized_res) == false) {
            this.it_ += 1;
            time.start();

            // === ADAPTIVE UPWINDING (Crovato thesis) ===
            // Start with strong stabilization (high MU_C, low MACH_C), then switch to
            // accurate values (low MU_C, high MACH_C) once residual drops below threshold
            if this.inputs_.ADAPTIVE_UPWIND_ && !this.inputs_.upwindAdapted_ {
                if normalized_res < this.inputs_.ADAPT_THRESHOLD_ {
                    this.spatialDisc_.inputs_.MU_C_ = this.inputs_.MU_C_FINAL_;
                    this.spatialDisc_.inputs_.MACH_C_ = this.inputs_.MACH_C_FINAL_;
                    this.inputs_.upwindAdapted_ = true;
                    writeln("  >>> Adaptive upwinding: switching to MU_C=", this.spatialDisc_.inputs_.MU_C_, ", MACH_C=", this.spatialDisc_.inputs_.MACH_C_);
                }
            }

            // === ADAPTIVE BETA (for Newton consistency / quadratic convergence) ===
            // Start with non-zero BETA for Jacobian conditioning, reduce to zero near
            // convergence to restore true Newton method with quadratic convergence rate
            if this.inputs_.ADAPTIVE_BETA_ && !this.inputs_.betaAdapted_ {
                if normalized_res < this.inputs_.BETA_THRESHOLD_ {
                    this.inputs_.BETA_ = this.inputs_.BETA_FINAL_;
                    this.inputs_.betaAdapted_ = true;
                    writeln("  >>> Adaptive BETA: switching to BETA=", this.inputs_.BETA_, " for Newton consistency");
                }
            }

            this.computeJacobian();
            
            // Always use full Newton step in RHS (omega will be applied to the solution update)
            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.b_petsc.set(elem-1, -this.spatialDisc_.res_[elem]);
            }
            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.b_petsc.set(this.spatialDisc_.nelemDomain_ + elem-1, -this.spatialDisc_.resPhi_[elem]);
            }
            forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
                this.b_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, -this.spatialDisc_.resWake_[i]);
            }
            this.b_petsc.assemblyComplete();

            // === PETSC GMRES ===
            const (its, reason) = GMRES(this.ksp, this.A_petsc, this.b_petsc, this.x_petsc);

            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.spatialDisc_.phi_[elem] += omega * this.x_petsc.get(elem-1);
                this.spatialDisc_.rhorho_[elem] += omega * this.x_petsc.get(this.spatialDisc_.nelemDomain_ + elem-1);
            }
            forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
                this.spatialDisc_.wakeFaceGamma_[i] += omega * this.x_petsc.get(2*this.spatialDisc_.nelemDomain_ + i -1);
            }
            
            // Compute residual for convergence check
            this.spatialDisc_.run();
            res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
            normalized_res = res / this.first_res_;

            resPhi = RMSE(this.spatialDisc_.resPhi_, this.spatialDisc_.elemVolume_);
            resWake = RMSE(this.spatialDisc_.resWake_);

            const res_wall = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wall_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wall_dom]);
            const res_fluid = RMSE(this.spatialDisc_.res_[this.spatialDisc_.fluid_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.fluid_dom]);
            const res_wake = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wake_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wake_dom]);
            const res_shock = RMSE(this.spatialDisc_.res_[this.spatialDisc_.shock_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.shock_dom]);

            const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
            time.stop();
            const elapsed = time.elapsed() + this.t0_;
            writeln(" Time: ", elapsed, " It: ", this.it_,
                    " res: ", res, " norm res: ", normalized_res, " phi res: ", resPhi, " wake res: ", resWake,
                    " res wall: ", res_wall, " res fluid: ", res_fluid, " res wake: ", res_wake, " res shock: ", res_shock,
                    " Cl: ", Cl, " Cd: ", Cd, " Cm: ", Cm, " Circulation: ", this.spatialDisc_.circulation_,
                    " GMRES its: ", its, " reason: ", reason, " omega: ", omega);
            

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
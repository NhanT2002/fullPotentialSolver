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

        // Add 1 extra DOF for circulation Γ (Kutta condition)
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
            // Pre-allocate circulation column for wake cells
            this.A_petsc.set(elem-1, this.gammaIndex_, 0.0);
        }
        // === KUTTA CONDITION ROW ===
        // K = Γ - (φ_upper - φ_lower) = 0
        // ∂K/∂Γ = 1
        // ∂K/∂φ_upper = -1
        // ∂K/∂φ_lower = +1
        this.A_petsc.set(this.gammaIndex_, this.gammaIndex_, 0.0);
        // Upper TE element
        this.A_petsc.set(this.gammaIndex_, this.spatialDisc_.upperTEelem_ - 1, 0.0);
        // Lower TE element
        this.A_petsc.set(this.gammaIndex_, this.spatialDisc_.lowerTEelem_ - 1, 0.0);
        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
    }

    // Number of wake faces with gamma coupled to circulation (for Jacobian)
    // This is set based on convection distance in initialize()
    var nCoupledWakeFaces_: int = 1;

    proc computeGradientSensitivity() {
        // Compute ∂(∇φ)/∂Γ for each cell.
        // This captures how each cell's gradient depends on circulation through
        // its wake-crossing faces.
        //
        // For UNSTEADY flow with vortex shedding:
        // Only the first nCoupledWakeFaces_ wake faces have their gamma coupled to circulation_.
        // Other wake faces have frozen gamma from previous time steps.
        //
        // For QUASI-STEADY mode (large TIME_STEP):
        // All wake faces have gamma = circulation_, so all contribute to sensitivity.
        //
        // For cell I with gradient: ∇φ_I = Σ_k w_Ik * (φ_k_corrected - φ_I)
        // where φ_k_corrected includes the Γ correction for wake-crossing neighbors:
        //   - If I above (1), k below (-1): φ_k_corrected = φ_k + Γ_face
        //   - If I below (-1), k above (1): φ_k_corrected = φ_k - Γ_face
        //
        // Therefore: ∂(∇φ_I)/∂Γ = Σ_{coupled faces} w_Ik * (±1)

        // Reset arrays
        this.dgradX_dGamma_ = 0.0;
        this.dgradY_dGamma_ = 0.0;

        const numWakeFaces = this.spatialDisc_.wake_face_dom.size;
        if numWakeFaces == 0 {
            return;
        }

        // Determine number of coupled wake faces based on convection distance
        var nCoupled = this.nCoupledWakeFaces_;
        if nCoupled > numWakeFaces {
            nCoupled = numWakeFaces;
        }

        // Process all coupled wake faces (first nCoupled faces, sorted by x from TE)
        for wakeIdx in 1..nCoupled {
            const face = this.spatialDisc_.wakeFace_[wakeIdx];
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];

            // Process elem1
            if elem1 <= this.spatialDisc_.nelemDomain_ {
                const kuttaType_elem = this.spatialDisc_.kuttaCell_[elem1];
                const kuttaType_neighbor = this.spatialDisc_.kuttaCell_[elem2];
                
                var wx = this.spatialDisc_.lsGradQR_!.wxFinal1_[face];
                var wy = this.spatialDisc_.lsGradQR_!.wyFinal1_[face];

                // Sign: +1 if elem above (1), neighbor below (-1)
                //       -1 if elem below (-1), neighbor above (1)
                const gammaSgn = if kuttaType_elem == 1 then 1.0 else -1.0;

                this.dgradX_dGamma_[elem1] += wx * gammaSgn;
                this.dgradY_dGamma_[elem1] += wy * gammaSgn;
            }

            // Process elem2
            if elem2 <= this.spatialDisc_.nelemDomain_ {
                const kuttaType_elem = this.spatialDisc_.kuttaCell_[elem2];
                const kuttaType_neighbor = this.spatialDisc_.kuttaCell_[elem1];
                
                var wx = this.spatialDisc_.lsGradQR_!.wxFinal2_[face];
                var wy = this.spatialDisc_.lsGradQR_!.wyFinal2_[face];

                // Sign: +1 if elem above (1), neighbor below (-1)
                //       -1 if elem below (-1), neighbor above (1)
                const gammaSgn = if kuttaType_elem == 1 then 1.0 else -1.0;

                this.dgradX_dGamma_[elem2] += wx * gammaSgn;
                this.dgradY_dGamma_[elem2] += wy * gammaSgn;
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
            
            // Time derivative contribution: d/dφ [ V * (ρ^{n+1} - ρ^n) / Δt ]
            // = V / Δt * dρ/dφ = V / Δt * (∂ρ/∂φ_t * dφ_t/dφ) = V / Δt * (-ρ^{2-γ} * M²) * (1/Δt)
            // = -V * ρ^{2-γ} * M² / Δt²
            var diag = -this.spatialDisc_.elemVolume_[elem] / (this.inputs_.TIME_STEP_**2) *
                        this.spatialDisc_.rhorho_[elem] ** (2.0 - this.spatialDisc_.inputs_.GAMMA_) *
                        this.spatialDisc_.inputs_.MACH_**2; // Time derivative contribution
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

                    // === Transonic DENSITY RETARDATION CONTRIBUTION ===
                    // In supersonic regions, ρ_face depends on upwind cell density:
                    //   ρ_face = (1-μ) * ρ_isen + μ * ρ_upwind = ρ_isen + μ*(ρ_upwind - ρ_isen)
                    // We consider ρ_isen and ρ_upwind constants here for simplicity.
                    // Thus, we will only compute contribution from μ dependence on φ.
                    // For interior faces, upwind depends on flow direction.
                    // So ∂F/∂φ includes: (V_avg · n) * ∂ρ_face/∂φ_upwind * A
                    const upwindElem = this.spatialDisc_.upwindElem_[face];
                    const mu_upwind = this.spatialDisc_.mumu_[upwindElem];

                    // if mu_upwind > 0.0 {
                    //     const uFace = this.spatialDisc_.uFace_[face];
                    //     const vFace = this.spatialDisc_.vFace_[face];
                    //     const V_dot_n = uFace * nx + vFace * ny;
                    //     const rho_isen = this.spatialDisc_.rhoIsenFace_[face];
                    //     const rho_upwind = this.spatialDisc_.rhorho_[upwindElem];
                    //     // Compute dμ/dφ_upwind
                    //     // μ = MU_C * (Mach^2 - MACH_C^2)
                    //     // Mach^2 = (u^2 + v^2) / a^2 // We consider a constant here for simplicity
                    //     // dμ/dφ = MU_C * d(Mach^2)/dφ
                    //     // d(Mach^2)/dφ = 2*(u*du/dφ + v*dv/dφ) / a^2
                    //     if elem == upwindElem {
                    //         // upwind is elem
                    //         const uElem = this.spatialDisc_.uu_[elem];
                    //         const vElem = this.spatialDisc_.vv_[elem];
                    //         const rhoElem = this.spatialDisc_.rhorho_[elem];
                    //         const a_squared = rhoElem**(this.spatialDisc_.inputs_.GAMMA_ - 1.0) / this.spatialDisc_.inputs_.MACH_**2;

                    //         const du_dphi = -sumWx_elem;
                    //         const dv_dphi = -sumWy_elem;

                    //         const dMach2_dphi = 2.0 * (uElem * du_dphi + vElem * dv_dphi) / a_squared;
                    //         const dmu_dphi = this.spatialDisc_.inputs_.MU_C_ * dMach2_dphi;

                    //         // Contribution to dflux/dφ_elem
                    //         const densityRetardationContrib = V_dot_n * dmu_dphi * (rho_upwind - rho_isen) * area;
                    //         diag += sign * densityRetardationContrib;

                    //         // Contribution to dflux/dφ_neighbor
                    //         const du_dphi_neighbor = wx_elemToNeighbor;
                    //         const dv_dphi_neighbor = wy_elemToNeighbor;
                    //         const dMach2_dphi_neighbor = 2.0 * (uElem * du_dphi_neighbor + vElem * dv_dphi_neighbor) / a_squared;
                    //         const dmu_dphi_neighbor = this.spatialDisc_.inputs_.MU_C_ * dMach2_dphi_neighbor;
                    //         const densityRetardationContrib_neighbor = V_dot_n * dmu_dphi_neighbor * (rho_upwind - rho_isen) * area;
                    //         offdiag += sign * densityRetardationContrib_neighbor;
                    //     } 
                    //     else if neighbor == upwindElem {
                    //         // upwind is neighbor
                    //         const uNeighbor = this.spatialDisc_.uu_[neighbor];
                    //         const vNeighbor = this.spatialDisc_.vv_[neighbor];
                    //         const rhoNeighbor = this.spatialDisc_.rhorho_[neighbor];
                    //         const a_squared = rhoNeighbor**(this.spatialDisc_.inputs_.GAMMA_ - 1.0) / this.spatialDisc_.inputs_.MACH_**2;

                    //         const du_dphi = wx_neighborToElem;
                    //         const dv_dphi = wy_neighborToElem;

                    //         const dMach2_dphi = 2.0 * (uNeighbor * du_dphi + vNeighbor * dv_dphi) / a_squared;
                    //         const dmu_dphi = this.spatialDisc_.inputs_.MU_C_ * dMach2_dphi;

                    //         // Contribution to dflux/dφ_elem
                    //         const densityRetardationContrib = V_dot_n * dmu_dphi * (rho_upwind - rho_isen) * area;
                    //         diag += sign * densityRetardationContrib;

                    //         // Contribution to dflux/dφ_neighbor
                    //         const du_dphi_neighbor = -sumWx_neighbor;
                    //         const dv_dphi_neighbor = -sumWy_neighbor;
                    //         const dMach2_dphi_neighbor = 2.0 * (uNeighbor * du_dphi_neighbor + vNeighbor * dv_dphi_neighbor) / a_squared;
                    //         const dmu_dphi_neighbor = this.spatialDisc_.inputs_.MU_C_ * dMach2_dphi_neighbor;
                    //         const densityRetardationContrib_neighbor = V_dot_n * dmu_dphi_neighbor * (rho_upwind - rho_isen) * area;
                    //         offdiag += sign * densityRetardationContrib_neighbor;
                    //     }
                    // }
                    
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

                    // Direct term: only for coupled wake faces
                    // For unsteady flow with vortex shedding, only the first nCoupledWakeFaces_
                    // wake faces have their gamma coupled to circulation_. Other wake faces have
                    // frozen gamma from previous time steps.
                    // directCoeff * ((φ_neighbor ± Γ) - φ_elem)
                    // ∂/∂Γ = ±directCoeff (sign depends on which side of wake)
                    if isWakeCrossingFace {
                        // Check if this face is in the coupled set
                        try {
                            const wakeIdx = this.spatialDisc_.wakeFace2index_[face];
                            if wakeIdx <= this.nCoupledWakeFaces_ {
                                var gammaSgn = 0.0;
                                if kuttaType_elem == 1 && kuttaType_neighbor == -1 {
                                    gammaSgn = 1.0;  // elem above, neighbor below: +Γ
                                } else {
                                    gammaSgn = -1.0; // elem below, neighbor above: -Γ
                                }
                                dFlux_dGamma += directCoeff * area * rhoFace * gammaSgn;
                            }
                        } catch e: Error {
                            // Face not in wake mapping - skip
                        }
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
                    // Similar to interior faces, but upwind cell can only be the interior cell
                    const upwindElem = elem;
                    const mu_upwind = this.spatialDisc_.mumu_[upwindElem];

                    // if mu_upwind > 0.0 {
                    //     const uFace = this.spatialDisc_.uFace_[face];
                    //     const vFace = this.spatialDisc_.vFace_[face];
                    //     const V_dot_n = uFace * nx + vFace * ny;
                    //     const rho_isen = this.spatialDisc_.rhoIsenFace_[face];
                    //     const rho_upwind = this.spatialDisc_.rhorho_[upwindElem];
                    //     // Compute dμ/dφ_upwind
                    //     const uElem = this.spatialDisc_.uu_[elem];
                    //     const vElem = this.spatialDisc_.vv_[elem];
                    //     const rhoElem = this.spatialDisc_.rhorho_[elem];
                    //     const a_squared = rhoElem**(this.spatialDisc_.inputs_.GAMMA_ - 1.0) / this.spatialDisc_.inputs_.MACH_**2;

                    //     const du_dphi = -sumWx_elem + wx_elemToNeighbor;
                    //     const dv_dphi = -sumWy_elem + wy_elemToNeighbor;

                    //     const dMach2_dphi = 2.0 * (uElem * du_dphi + vElem * dv_dphi) / a_squared;
                    //     const dmu_dphi = this.spatialDisc_.inputs_.MU_C_ * dMach2_dphi;

                    //     // Contribution to dflux/dφ_elem
                    //     const densityRetardationContrib = V_dot_n * dmu_dphi * (rho_upwind - rho_isen) * area;
                    //     diag += sign * densityRetardationContrib;
                    // }

                    // === CIRCULATION (Γ) DERIVATIVE FOR WALL FACES ===
                    // Wall flux = V_int · m_wall * A, where V_int = ∇φ_int
                    // If the interior cell has wake-crossing faces, ∇φ_int depends on Γ
                    // ∂flux/∂Γ = (∂∇φ_int/∂Γ · m_wall) * A
                    const dgradX_elem_wall = this.dgradX_dGamma_[elem];
                    const dgradY_elem_wall = this.dgradY_dGamma_[elem];
                    const dFlux_dGamma_wall = (dgradX_elem_wall * mWallX + dgradY_elem_wall * mWallY) * sign * area * rhoFace;
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
                
                if machFace >= this.spatialDisc_.inputs_.MACH_C_ {
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
                            * this.spatialDisc_.invL_IJ_[face];
                            
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
        this.A_petsc.add(this.gammaIndex_, this.gammaIndex_, 1.0);
        
        // dKutta/dφ_upperTE = -1.0
        this.A_petsc.add(this.gammaIndex_, this.spatialDisc_.upperTEelem_ - 1, -1.0);
        // dKutta/dφ_lowerTE = +1.0
        this.A_petsc.add(this.gammaIndex_, this.spatialDisc_.lowerTEelem_ - 1, 1.0);
        
        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
    }

    proc initialize() {
        this.spatialDisc_.initializeMetrics();
        this.spatialDisc_.initializeKuttaCells();
        this.spatialDisc_.initializeSolution();
        this.spatialDisc_.run();
        this.initializeJacobian();
        this.computeGradientSensitivity();
        
        this.computeJacobian();

        // if this.inputs_.START_FILENAME_ != "" {
        //     writeln("Initializing solution from file: ", this.inputs_.START_FILENAME_);
        //     const (xElem, yElem, rho, phi, it, time, res, cl, cd, cm, circulation, wakeGamma) = readSolution(this.inputs_.START_FILENAME_);
        //     for i in it.domain {
        //         this.timeList_.pushBack(time[i]);
        //         this.itList_.pushBack(it[i]);
        //         this.resList_.pushBack(res[i]);
        //         this.clList_.pushBack(cl[i]);
        //         this.cdList_.pushBack(cd[i]);
        //         this.cmList_.pushBack(cm[i]);
        //         this.circulationList_.pushBack(circulation[i]);
        //     }
        //     this.it_ = it.last;
        //     this.t0_ = time.last;
        //     this.first_res_ = res.first;
        // }

        // Reset adaptive upwinding state for new Mach number
        this.inputs_.upwindAdapted_ = false;
        if this.inputs_.ADAPTIVE_UPWIND_ {
            this.spatialDisc_.inputs_.MU_C_ = this.inputs_.MU_C_START_;
            this.spatialDisc_.inputs_.MACH_C_ = this.inputs_.MACH_C_START_;
            writeln("  >>> Adaptive upwinding: starting with MU_C=", this.inputs_.MU_C_, ", MACH_C=", this.inputs_.MACH_C_);
        }
        
        // Calculate number of coupled wake faces based on convection distance
        // Convection distance per time step: Δx = U_∞ * Δt = Δt (non-dimensional)
        const numWakeFaces = this.spatialDisc_.wake_face_dom.size;
        if numWakeFaces > 1 {
            const wakeLength = this.spatialDisc_.wakeFaceX_[numWakeFaces] - this.spatialDisc_.wakeFaceX_[1];
            const avgSpacing = wakeLength / (numWakeFaces - 1);
            const convectionDistance = this.inputs_.TIME_STEP_;  // U_INF = 1.0 non-dimensional
            
            // Number of wake faces with gamma coupled to circulation
            this.nCoupledWakeFaces_ = (convectionDistance / avgSpacing): int;
            if this.nCoupledWakeFaces_ < 1 {
                this.nCoupledWakeFaces_ = 1;
            }
            if this.nCoupledWakeFaces_ >= numWakeFaces {
                this.nCoupledWakeFaces_ = numWakeFaces;
                writeln("  >>> Quasi-steady mode: convection distance (", convectionDistance, 
                        ") >= wake length (", wakeLength, "), all ", numWakeFaces, " wake faces coupled");
                // Reset all wake gamma to current circulation
                this.spatialDisc_.wakeFaceGamma_ = this.spatialDisc_.circulation_;
                this.spatialDisc_.run();  // Recompute with updated wake gamma
            } else {
                writeln("  >>> Unsteady mode: ", this.nCoupledWakeFaces_, " of ", numWakeFaces, 
                        " wake faces coupled (convection distance = ", convectionDistance, 
                        ", avg spacing = ", avgSpacing, ")");
            }
        } else {
            this.nCoupledWakeFaces_ = numWakeFaces;
        }
        
        // Recompute gradient sensitivity with correct number of coupled faces
        this.computeGradientSensitivity();
        
        // Reset adaptive BETA state for new Mach number
        this.inputs_.betaAdapted_ = false;
        if this.inputs_.ADAPTIVE_BETA_ {
            this.inputs_.BETA_ = this.inputs_.BETA_START_;
            writeln("  >>> Adaptive BETA: starting with BETA=", this.inputs_.BETA_);
        }
    }

    /**
     * Main driver for unsteady simulation with oscillating alpha.
     * This is the outer loop that advances through physical time steps.
     * At each time step:
     *   1. Update alpha based on oscillation formula
     *   2. Store current phi as phi_m1_ (for BDF1)
     *   3. Advance wake vortices (shift gamma values downstream)
     *   4. Call Newton solver to converge this time step
     *   5. Output results
     */
    proc solveUnsteady() {
        var physicalTime: real(64) = 0.0;
        var timeStepCount: int = 0;
        const dt = this.inputs_.TIME_STEP_;
        const tFinal = this.inputs_.TIME_FINAL_;
        
        // Lists to store unsteady history at each physical time step
        var unsteadyTimeList = new list(real(64));
        var unsteadyStepList = new list(int);
        var unsteadyAlphaList = new list(real(64));
        var unsteadyClList = new list(real(64));
        var unsteadyCdList = new list(real(64));
        var unsteadyCmList = new list(real(64));
        var unsteadyCirculationList = new list(real(64));
        
        // Oscillating alpha parameters
        const alpha0 = this.inputs_.ALPHA_0_;           // Mean angle of attack [degrees]
        const alphaAmp = this.inputs_.ALPHA_AMPLITUDE_; // Oscillation amplitude [degrees]
        const alphaFreq = this.inputs_.ALPHA_FREQUENCY_;// Reduced frequency k = ω*c/(2*U_inf)
        const alphaPhase = this.inputs_.ALPHA_PHASE_;   // Phase offset [radians]
        
        writeln("\n=== UNSTEADY SIMULATION ===");
        writeln("  Time step: ", dt);
        writeln("  Final time: ", tFinal);
        writeln("  Alpha_0: ", alpha0, " deg");
        writeln("  Alpha amplitude: ", alphaAmp, " deg");
        writeln("  Reduced frequency k: ", alphaFreq);
        writeln("  Phase: ", alphaPhase, " rad");
        
        // Set initial alpha
        var alpha_current = alpha0 + alphaAmp * sin(alphaFreq * physicalTime + alphaPhase);
        this.inputs_.setAlpha(alpha_current);
        this.spatialDisc_.inputs_.setAlpha(alpha_current);
        writeln("  Initial alpha: ", alpha_current, " deg");
        writeln("");
        
        // Initialize and solve the first time step (from initial condition)
        this.initialize();
        this.solve();
        
        // Store initial state
        const (Cl0, Cd0, Cm0) = this.spatialDisc_.computeAerodynamicCoefficients();
        unsteadyTimeList.pushBack(physicalTime);
        unsteadyStepList.pushBack(0);
        unsteadyAlphaList.pushBack(alpha_current);
        unsteadyClList.pushBack(Cl0);
        unsteadyCdList.pushBack(Cd0);
        unsteadyCmList.pushBack(Cm0);
        unsteadyCirculationList.pushBack(this.spatialDisc_.circulation_);
        
        // Time stepping loop
        while physicalTime < tFinal {
            timeStepCount += 1;
            physicalTime += dt;
            
            writeln("\n=== TIME STEP ", timeStepCount, " | t = ", physicalTime, " ===");
            
            // 1. Update alpha based on oscillation
            alpha_current = alpha0 + alphaAmp * sin(alphaFreq * physicalTime + alphaPhase);
            this.inputs_.setAlpha(alpha_current);
            this.spatialDisc_.inputs_.setAlpha(alpha_current);
            writeln("  Alpha = ", alpha_current, " deg");
            
            // 2. Store current solution as phi_m1_ and rhorho_m1_ for BDF1 time derivative
            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.spatialDisc_.phi_m1_[elem] = this.spatialDisc_.phi_[elem];
                this.spatialDisc_.rhorho_m1_[elem] = this.spatialDisc_.rhorho_[elem];
            }
            
            // 3. Advance wake vortices (shift gamma values downstream)
            this.spatialDisc_.advanceWakeVortices();
            
            // 4. Reset iteration counter and convergence flags for new time step
            this.it_ = 0;
            this.inputs_.betaAdapted_ = false;
            if this.inputs_.ADAPTIVE_BETA_ {
                this.inputs_.BETA_ = this.inputs_.BETA_START_;
            }
            
            // 5. Recompute residual with new alpha and wake gamma
            this.spatialDisc_.run();
            
            // 6. Recompute gradient sensitivity (may have changed with nCoupledWakeFaces)
            this.computeGradientSensitivity();
            
            // 7. Solve Newton iterations for this time step
            this.solve();
            
            // Output progress and store unsteady history
            const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
            writeln("  Time step ", timeStepCount, " converged: Cl=", Cl, ", Cd=", Cd, 
                    ", Gamma=", this.spatialDisc_.circulation_);
            
            // Store time step results
            unsteadyTimeList.pushBack(physicalTime);
            unsteadyStepList.pushBack(timeStepCount);
            unsteadyAlphaList.pushBack(alpha_current);
            unsteadyClList.pushBack(Cl);
            unsteadyCdList.pushBack(Cd);
            unsteadyCmList.pushBack(Cm);
            unsteadyCirculationList.pushBack(this.spatialDisc_.circulation_);
        }
        
        writeln("\n=== SIMULATION COMPLETE ===");
        writeln("  Total time steps: ", timeStepCount);
        writeln("  Final physical time: ", physicalTime);
        
        // Write final solution with unsteady history
        this.spatialDisc_.writeSolutionUnsteady(
            this.timeList_, this.itList_, this.resList_, this.clList_, this.cdList_, this.cmList_, this.circulationList_,
            unsteadyTimeList, unsteadyStepList, unsteadyAlphaList, unsteadyClList, unsteadyCdList, unsteadyCmList
        );
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

        // Initial residual
        res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
        this.first_res_ = res;
        normalized_res = res / this.first_res_;
        const res_wall = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wall_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wall_dom]);
        const res_fluid = RMSE(this.spatialDisc_.res_[this.spatialDisc_.fluid_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.fluid_dom]);
        const res_wake = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wake_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wake_dom]);
        const res_shock = RMSE(this.spatialDisc_.res_[this.spatialDisc_.shock_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.shock_dom]);
        const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
        const elapsed = this.t0_;
        writeln(" Time: ", elapsed, " It: ", this.it_,
                " res: ", res, " norm res: ", normalized_res, " kutta res: ", this.spatialDisc_.kutta_res_,
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
            // Add Kutta condition residual to RHS
            this.b_petsc.set(this.gammaIndex_, -this.spatialDisc_.kutta_res_);
            this.b_petsc.assemblyComplete();

            // === PETSC GMRES ===
            const (its, reason) = GMRES(this.ksp, this.A_petsc, this.b_petsc, this.x_petsc);
            
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
                    // Update circulation
                    this.spatialDisc_.circulation_ = circulation_backup + omega * this.x_petsc.get(this.gammaIndex_);
                    
                    // Update ALL coupled wake faces to current circulation
                    // In quasi-steady mode, all faces are coupled; in unsteady mode, only the first nCoupledWakeFaces_
                    if this.spatialDisc_.wake_face_dom.size > 0 {
                        for i in 1..this.nCoupledWakeFaces_ {
                            this.spatialDisc_.wakeFaceGamma_[i] = this.spatialDisc_.circulation_;
                        }
                    }
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
                // Update circulation
                this.spatialDisc_.circulation_ += omega * this.x_petsc.get(this.gammaIndex_);
                // Update ALL coupled wake faces to current circulation
                if this.spatialDisc_.wake_face_dom.size > 0 {
                    for i in 1..this.nCoupledWakeFaces_ {
                        this.spatialDisc_.wakeFaceGamma_[i] = this.spatialDisc_.circulation_;
                    }
                }
                
                // Compute residual for convergence check
                this.spatialDisc_.run();
                res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
            }
            
            res_prev = res;
            normalized_res = res / this.first_res_;

            const res_wall = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wall_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wall_dom]);
            const res_fluid = RMSE(this.spatialDisc_.res_[this.spatialDisc_.fluid_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.fluid_dom]);
            const res_wake = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wake_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wake_dom]);
            const res_shock = RMSE(this.spatialDisc_.res_[this.spatialDisc_.shock_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.shock_dom]);

            const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
            time.stop();
            const elapsed = time.elapsed() + this.t0_;
            writeln(" Time: ", elapsed, " It: ", this.it_,
                    " res: ", res, " norm res: ", normalized_res, " kutta res: ", this.spatialDisc_.kutta_res_,
                    " res wall: ", res_wall, " res fluid: ", res_fluid, " res wake: ", res_wake, " res shock: ", res_shock,
                    " Cl: ", Cl, " Cd: ", Cd, " Cm: ", Cm, " Circulation: ", this.spatialDisc_.circulation_,
                    " GMRES its: ", its, " reason: ", reason, " omega: ", omega, " LS its: ", lineSearchIts);
            

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
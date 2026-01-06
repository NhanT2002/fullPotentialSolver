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

        const nWakeFace = this.spatialDisc_.wakeFace_.size
        const M = spatialDisc.nelemDomain_*2 + nWakeFace;
        const N = spatialDisc.nelemDomain_*2 + nWakeFace;
        this.gammaIndex_ = spatialDisc.nelemDomain_*2;  // 0-based index for Γ
        
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
            const kuttaType1 = this.spatialDisc_.kuttaCell_[elem1];
            const kuttaType2 = this.spatialDisc_.kuttaCell_[elem2];
            if (kuttaType1 == 1 && kuttaType2 == -1) || (kuttaType1 == -1 && kuttaType2 == 1) {
                this.A_petsc.set(elem1-1, 2*this.spatialDisc_.nelemDomain_ + i -1, 0.0);
                this.A_petsc.set(elem2-1, 2*this.spatialDisc_.nelemDomain_ + i -1, 0.0);
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
            const kuttaType1 = this.spatialDisc_.kuttaCell_[elem1];
            const kuttaType2 = this.spatialDisc_.kuttaCell_[elem2];
            if ( (kuttaType1 == 1 && kuttaType2 == -1) || (kuttaType1 == -1 && kuttaType2 == 1) ) {
                this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem1-1, 2*this.spatialDisc_.nelemDomain_ + i -1, 0.0);
                this.A_petsc.set(this.spatialDisc_.nelemDomain_ + elem2-1, 2*this.spatialDisc_.nelemDomain_ + i -1, 0.0);
            }
        }

        // seventh block: dRes^(wake)/d(phi)
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            const elem1 = this.spatialDisc_.mesh_.edge2elem_[1, face];
            const elem2 = this.spatialDisc_.mesh_.edge2elem_[2, face];
            // dRes^(wake)/d(phi_elem1)
            this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, this.spatialDisc_.nelemDomain_ + elem1 -1, 0.0);
            // dRes^(wake)/d(phi_elem2)
            this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, this.spatialDisc_.nelemDomain_ + elem2 -1, 0.0);
        }
        // eighth block: dRes^(wake)/d(Γ)
        forall (i, face) in zip(this.spatialDisc_.wake_face_dom, this.spatialDisc_.wakeFace_) {
            // dRes^(wake)/d(Γ_i) = 1
            this.A_petsc.set(2*this.spatialDisc_.nelemDomain_ + i -1, 2*this.spatialDisc_.nelemDomain_ + i -1, 1.0);
        }

        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
    }

    proc computeJacobian() {
        

        this.A_petsc.zeroEntries();
        
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

            // === PETSC GMRES ===
            const (petscIts, petscReason) = GMRES(this.ksp, this.A_petsc, this.b_petsc, this.x_petsc);
            its = petscIts;
            reason = petscReason;
            
            
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
            
            res_prev = res;
            normalized_res = res / this.first_res_;

            const res_wall = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wall_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wall_dom]);
            const res_fluid = RMSE(this.spatialDisc_.res_[this.spatialDisc_.fluid_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.fluid_dom]);
            const res_wake = RMSE(this.spatialDisc_.res_[this.spatialDisc_.wake_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.wake_dom]);
            const res_shock = RMSE(this.spatialDisc_.res_[this.spatialDisc_.shock_dom], this.spatialDisc_.elemVolume_[this.spatialDisc_.shock_dom]);

            const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
            time.stop();
            const elapsed = time.elapsed() + this.t0_;
            if this.inputs_.ADAPTIVE_GMRES_TOL_ {
                writeln(" Time: ", elapsed, " It: ", this.it_,
                        " res: ", res, " norm res: ", normalized_res, " kutta res: ", this.spatialDisc_.kutta_res_,
                        " res wall: ", res_wall, " res fluid: ", res_fluid, " res wake: ", res_wake, " res shock: ", res_shock,
                        " Cl: ", Cl, " Cd: ", Cd, " Cm: ", Cm, " Circulation: ", this.spatialDisc_.circulation_,
                        " GMRES its: ", its, " reason: ", reason, " omega: ", omega, " LS its: ", lineSearchIts,
                        " GMRES rtol: ", current_rtol);
            } else {
                writeln(" Time: ", elapsed, " It: ", this.it_,
                        " res: ", res, " norm res: ", normalized_res, " kutta res: ", this.spatialDisc_.kutta_res_,
                        " res wall: ", res_wall, " res fluid: ", res_fluid, " res wake: ", res_wake, " res shock: ", res_shock,
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
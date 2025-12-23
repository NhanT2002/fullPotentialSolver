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

    proc init(spatialDisc: shared spatialDiscretization, ref inputs: potentialInputs) {
        writeln("Initializing temporal discretization...");
        this.spatialDisc_ = spatialDisc;
        this.inputs_ = inputs;

        const M = spatialDisc.nelemDomain_;
        const N = spatialDisc.nelemDomain_;
        this.A_petsc = new owned PETSCmatrix_c(PETSC_COMM_SELF, "seqaij", M, M, N, N);
        this.x_petsc = new owned PETSCvector_c(PETSC_COMM_SELF, N, N, 0.0, "seq");
        this.b_petsc = new owned PETSCvector_c(PETSC_COMM_SELF, N, N, 0.0, "seq");

        var nnz : [0..M-1] PetscInt;
        nnz = 4*(this.spatialDisc_.mesh_.elem2edge_[this.spatialDisc_.mesh_.elem2edgeIndex_[1] + 1 .. this.spatialDisc_.mesh_.elem2edgeIndex_[1 + 1]].size + 1);
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
        }

        this.A_petsc.assemblyComplete();
        // this.A_petsc.matView();
    }

    proc computeJacobian() {
        // Compute the Jacobian matrix d(res_I)/d(phi_J) for the linear system.
        //
        // Residual: res_I = sum over faces f of elem I: sign_f * flux_f
        //
        // Simplified flux at face (between I and J), ignoring correction term:
        //   flux = (V_face · n) * A
        //   V_face = V_avg = 0.5 * (gradPhi_I + gradPhi_J)
        //
        // So:
        //   V_face · n = 0.5 * (gradPhi_I · n + gradPhi_J · n)
        //              = 0.5 * (u_I*nx + v_I*ny + u_J*nx + v_J*ny)
        //
        // Since gradPhi_I = sum_k w_Ik * (phi_k - phi_I):
        //   d(gradPhi_I)/d(phi_I) = -sumW_I (sum of all weights)
        //   d(gradPhi_I)/d(phi_k) = w_Ik (weight for neighbor k)
        //
        // Jacobian contributions for face IJ with sign s (from elem I's perspective):
        //   d(res_I)/d(phi_I) += s * A * 0.5 * (-sumWx_I*nx - sumWy_I*ny)
        //   d(res_I)/d(phi_J) += s * A * 0.5 * (wx_IJ*nx + wy_IJ*ny - sumWx_J*nx - sumWy_J*ny)
        
        this.A_petsc.zeroEntries();
        
        forall elem in 1..this.spatialDisc_.nelemDomain_ {
            const faces = this.spatialDisc_.mesh_.elem2edge_[
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem] + 1 .. 
                this.spatialDisc_.mesh_.elem2edgeIndex_[elem + 1]];
            
            var diag = 0.0;  // Diagonal contribution d(res_elem)/d(phi_elem)
            
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
                
                // Get gradient weights for this element
                var sumWx_elem: real(64);
                var sumWy_elem: real(64);
                var wx_elemToNeighbor: real(64);
                var wy_elemToNeighbor: real(64);
                
                if elem1 == elem {
                    // elem is elem1, using weights from perspective 1
                    sumWx_elem = this.spatialDisc_.lsGradQR_!.sumWx1_[elem];
                    sumWy_elem = this.spatialDisc_.lsGradQR_!.sumWy1_[elem];
                    wx_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wxFinal1_[face];
                    wy_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wyFinal1_[face];
                } else {
                    // elem is elem2, using weights from perspective 2
                    sumWx_elem = this.spatialDisc_.lsGradQR_!.sumWx2_[elem];
                    sumWy_elem = this.spatialDisc_.lsGradQR_!.sumWy2_[elem];
                    wx_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wxFinal2_[face];
                    wy_elemToNeighbor = this.spatialDisc_.lsGradQR_!.wyFinal2_[face];
                }
                
                // Diagonal contribution from d(V_face · n)/d(phi_elem):
                // From d(gradPhi_elem)/d(phi_elem) = -sumW_elem
                // Contribution: 0.5 * (-sumWx*nx - sumWy*ny)
                const diag_contrib = 0.5 * (-sumWx_elem * nx - sumWy_elem * ny);
                diag += sign * diag_contrib * area;
                
                // Off-diagonal contribution from d(res_elem)/d(phi_neighbor):
                // 1. From d(gradPhi_elem)/d(phi_neighbor) = w_elemToNeighbor
                //    Contribution: 0.5 * (wx*nx + wy*ny)
                // 2. From d(gradPhi_neighbor)/d(phi_neighbor) = -sumW_neighbor
                //    Contribution: 0.5 * (-sumWx_neigh*nx - sumWy_neigh*ny)
                
                var offdiag = 0.0;
                
                // Contribution from gradient of elem using neighbor
                offdiag += 0.5 * (wx_elemToNeighbor * nx + wy_elemToNeighbor * ny);
                
                // Contribution from gradient of neighbor (need neighbor's sum weights)
                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    var sumWx_neighbor: real(64);
                    var sumWy_neighbor: real(64);
                    
                    if elem1 == neighbor {
                        sumWx_neighbor = this.spatialDisc_.lsGradQR_!.sumWx1_[neighbor];
                        sumWy_neighbor = this.spatialDisc_.lsGradQR_!.sumWy1_[neighbor];
                    } else {
                        sumWx_neighbor = this.spatialDisc_.lsGradQR_!.sumWx2_[neighbor];
                        sumWy_neighbor = this.spatialDisc_.lsGradQR_!.sumWy2_[neighbor];
                    }
                    offdiag += 0.5 * (-sumWx_neighbor * nx - sumWy_neighbor * ny);
                }
                
                // Apply sign and area
                offdiag *= sign * area;
                
                // Add to matrix (only for interior neighbors)
                if neighbor <= this.spatialDisc_.nelemDomain_ {
                    this.A_petsc.add(elem-1, neighbor-1, offdiag);
                }
            }
            
            // Add diagonal entry
            this.A_petsc.add(elem-1, elem-1, diag);
        }
        
        this.A_petsc.assemblyComplete();
        this.A_petsc.matView();
    }

    proc initialize() {
        this.spatialDisc_.initializeMetrics();
        this.spatialDisc_.initializeSolution();
        this.initializeJacobian();
    }

    proc solve() {
        var res: real(64) = 1e12;
        var first_res : real(64) = 1e12;
        var time: stopwatch;
        while (res > this.inputs_.CONV_TOL_ && this.it_ < this.inputs_.IT_MAX_ && isNan(res) == false) {
            this.it_ += 1;
            time.start();

            this.spatialDisc_.run();
            this.computeJacobian();
            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.b_petsc.set(elem-1, -this.inputs_.OMEGA_ * this.spatialDisc_.res_[elem]);
            }
            this.b_petsc.assemblyComplete();

            GMRES(this.ksp, this.A_petsc, this.b_petsc, this.x_petsc);

            forall elem in 1..this.spatialDisc_.nelemDomain_ {
                this.spatialDisc_.phi_[elem] += this.x_petsc.get(elem-1);
            }

            res = RMSE(this.spatialDisc_.res_, this.spatialDisc_.elemVolume_);
            if this.it_ == 1 {
                first_res = res;
            }

            const (Cl, Cd, Cm) = this.spatialDisc_.computeAerodynamicCoefficients();
            time.stop();
            const elapsed = time.elapsed();
            writeln(" Time: ", elapsed, " It: ", this.it_,
                    " Residual: ", res, " normalized Residual: ", res / first_res,
                    " Cl: ", Cl, " Cd: ", Cd, " Cm: ", Cm);

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
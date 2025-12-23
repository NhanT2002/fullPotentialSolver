module spatialDiscretization 
{
use mesh;
use writeCGNS;
use Math;
use List;
use Map;
use Set;
use IO;
use linearAlgebra;
use Time;
use Sort;
use leastSquaresGradient;
import input.potentialInputs;

config const elemID : int = 0;   // Element index for debugging purposes

class spatialDiscretization {
    var mesh_: shared MeshData;
    var inputs_: potentialInputs;
    var nelem_: int;
    var nelemDomain_ : int;
    var nface_: int;

    var elemDomain_dom: domain(1) = {1..0};
    var res_: [elemDomain_dom] real(64);

    var elem_dom: domain(1) = {1..0};
    var uu_: [elem_dom] real(64);
    var vv_: [elem_dom] real(64);
    var pp_: [elem_dom] real(64);
    var rhorho_: [elem_dom] real(64);
    var phi_: [elem_dom] real(64);
    var elemCentroidX_: [elem_dom] real(64);
    var elemCentroidY_: [elem_dom] real(64);
    var elemVolume_: [elem_dom] real(64);

    var face_dom: domain(1) = {1..0};
    var faceCentroidX_: [face_dom] real(64);
    var faceCentroidY_: [face_dom] real(64);
    var faceArea_: [face_dom] real(64);
    var faceNormalX_: [face_dom] real(64);
    var faceNormalY_: [face_dom] real(64);
    var uFace_: [face_dom] real(64);
    var vFace_: [face_dom] real(64);
    var rhoFace_: [face_dom] real(64);
    var pFace_: [face_dom] real(64);

    // Precomputed coefficients for flux computation (mesh-dependent only)
    var invL_IJ_: [face_dom] real(64);      // 1 / distance between cell centroids
    var t_IJ_x_: [face_dom] real(64);       // Unit vector from elem1 to elem2 (x-component)
    var t_IJ_y_: [face_dom] real(64);       // Unit vector from elem1 to elem2 (y-component)
    var corrCoeffX_: [face_dom] real(64);   // nx / (n · t_IJ) - correction coefficient (x)
    var corrCoeffY_: [face_dom] real(64);   // ny / (n · t_IJ) - correction coefficient (y)

    var faceFluxX_: [face_dom] real(64);   // Flux storage for Green-Gauss
    var faceFluxY_: [face_dom] real(64);
    var flux_: [face_dom] real(64);       // Flux storage for residual computation

    var gamma_minus_one_over_two_: real(64);
    var one_over_gamma_minus_one_: real(64);

    // Least-squares gradient operator (precomputed coefficients)
    var lsGrad_: owned LeastSquaresGradient?;
    var lsGradQR_: owned LeastSquaresGradientQR?;

    var circulation_ : real(64);

    proc init(Mesh: shared MeshData, ref inputs: potentialInputs) {
        this.mesh_ = Mesh;
        this.inputs_ = inputs;
        this.nelem_ = this.mesh_.nelemWithGhost_;
        this.nelemDomain_ = this.mesh_.nelem_;
        this.nface_ = this.mesh_.nedge_;

        this.elemDomain_dom = {1..this.nelemDomain_};

        this.elem_dom = {1..this.nelem_};
        
        this.face_dom = {1..this.nface_};

        this.gamma_minus_one_over_two_ = (this.inputs_.GAMMA_ - 1.0) / 2.0;
        this.one_over_gamma_minus_one_ = 1.0 / (this.inputs_.GAMMA_ - 1.0);
    }

    proc initializeMetrics() {
        // Compute element centroids and volumes in a single pass
        forall elem in 1..this.nelemDomain_ {
            const nodeStart = this.mesh_.elem2nodeIndex_[elem] + 1;
            const nodeEnd = this.mesh_.elem2nodeIndex_[elem + 1];
            const nodes = this.mesh_.elem2node_[nodeStart..nodeEnd];
            
            // Compute centroid
            var cx = 0.0, cy = 0.0;
            for node in nodes {
                cx += this.mesh_.X_[node];
                cy += this.mesh_.Y_[node];
            }
            const invN = 1.0 / nodes.size : real(64);
            cx *= invN;
            cy *= invN;
            this.elemCentroidX_[elem] = cx;
            this.elemCentroidY_[elem] = cy;
            
            // Compute volume using edges and centroid
            const edgeStart = this.mesh_.elem2edgeIndex_[elem] + 1;
            const edgeEnd = this.mesh_.elem2edgeIndex_[elem + 1];
            const edges = this.mesh_.elem2edge_[edgeStart..edgeEnd];
            
            var vol = 0.0;
            for edge in edges {
                const n1 = this.mesh_.edge2node_[1, edge];
                const n2 = this.mesh_.edge2node_[2, edge];
                const x1 = this.mesh_.X_[n1];
                const y1 = this.mesh_.Y_[n1];
                const x2 = this.mesh_.X_[n2];
                const y2 = this.mesh_.Y_[n2];
                vol += 0.5 * abs((x1-x2)*(y1+y2) + (x2-cx)*(y2+cy) + (cx-x1)*(cy+y1));
            }
            this.elemVolume_[elem] = vol;
        }

        // Compute ghost cell centroids by mirroring across boundary faces
        inline proc computeGhostCentroid(face: int) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            const node1 = this.mesh_.edge2node_[1, face];
            const node2 = this.mesh_.edge2node_[2, face];
            
            const x1 = this.mesh_.X_[node1];
            const y1 = this.mesh_.Y_[node1];
            const x2 = this.mesh_.X_[node2];
            const y2 = this.mesh_.Y_[node2];
            
            var A = y2-y1;
            var B = -(x2-x1);
            var C = -A*x1 - B*y1;
            
            const M = sqrt(A*A + B*B);

            A = A/M;
            B = B/M;
            C = C/M;

            const x = this.elemCentroidX_[interiorElem];
            const y = this.elemCentroidY_[interiorElem];

            const D = A*x + B*y + C;

            const x_mirror = x - 2*A*D;
            const y_mirror = y - 2*B*D;
            
            this.elemCentroidX_[ghostElem] = x_mirror;
            this.elemCentroidY_[ghostElem] = y_mirror;
        }
        
        forall face in this.mesh_.edgeWall_ do computeGhostCentroid(face);
        forall face in this.mesh_.edgeFarfield_ do computeGhostCentroid(face);

        // Compute face centroids, areas, and normals in a single pass
        forall face in 1..this.nface_ {
            const node1 = this.mesh_.edge2node_[1, face];
            const node2 = this.mesh_.edge2node_[2, face];
            
            const x1 = this.mesh_.X_[node1];
            const y1 = this.mesh_.Y_[node1];
            const x2 = this.mesh_.X_[node2];
            const y2 = this.mesh_.Y_[node2];
            
            const dx = x2 - x1;
            const dy = y2 - y1;
            const d = sqrt(dx*dx + dy*dy);
            
            // Face centroid and area
            const fcx = (x1 + x2) * 0.5;
            const fcy = (y1 + y2) * 0.5;
            this.faceCentroidX_[face] = fcx;
            this.faceCentroidY_[face] = fcy;
            this.faceArea_[face] = d;
            
            // Face normal (perpendicular to edge)
            var nx = dy / d;
            var ny = -dx / d;
            
            // Ensure normal points FROM elem1 TO elem2
            const elem1 = this.mesh_.edge2elem_[1, face];
            const toCentroidX = this.elemCentroidX_[elem1] - fcx;
            const toCentroidY = this.elemCentroidY_[elem1] - fcy;
            
            if (nx * toCentroidX + ny * toCentroidY > 0) {
                nx = -nx;
                ny = -ny;
            }
            
            this.faceNormalX_[face] = nx;
            this.faceNormalY_[face] = ny;
        }

        // Precompute flux coefficients (depends on mesh geometry only)
        forall face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Cell-to-cell vector
            const dx_IJ = this.elemCentroidX_[elem2] - this.elemCentroidX_[elem1];
            const dy_IJ = this.elemCentroidY_[elem2] - this.elemCentroidY_[elem1];
            const l_IJ = sqrt(dx_IJ*dx_IJ + dy_IJ*dy_IJ);
            
            // Store inverse distance
            this.invL_IJ_[face] = 1.0 / l_IJ;
            
            // Unit vector from elem1 to elem2
            this.t_IJ_x_[face] = dx_IJ / l_IJ;
            this.t_IJ_y_[face] = dy_IJ / l_IJ;
            
            // Correction coefficients: n / (n · t_IJ)
            const nx = this.faceNormalX_[face];
            const ny = this.faceNormalY_[face];
            const nDotT = nx * this.t_IJ_x_[face] + ny * this.t_IJ_y_[face];
            const invNDotT = 1.0 / nDotT;
            
            this.corrCoeffX_[face] = nx * invNDotT;
            this.corrCoeffY_[face] = ny * invNDotT;
        }

        // // Initialize and precompute least-squares gradient coefficients
        // this.lsGrad_ = new owned LeastSquaresGradient(this.mesh_, this.elemCentroidX_, this.elemCentroidY_);
        // this.lsGrad_!.precompute(this.elemCentroidX_, this.elemCentroidY_);

        // Initialize and precompute QR-based least-squares gradient (Blazek formulation)
        this.lsGradQR_ = new owned LeastSquaresGradientQR(this.mesh_, this.elemCentroidX_, this.elemCentroidY_);
        this.lsGradQR_!.precompute(this.elemCentroidX_, this.elemCentroidY_);
    }

    proc initializeSolution() {
        forall elem in 1..this.nelem_ {
            this.uu_[elem] = this.inputs_.U_INF_;
            this.vv_[elem] = this.inputs_.V_INF_;
            this.rhorho_[elem] = this.inputs_.RHO_INF_;
            this.pp_[elem] = this.inputs_.P_INF_;

            this.phi_[elem] = this.inputs_.U_INF_ * this.elemCentroidX_[elem] +
                              this.inputs_.V_INF_ * this.elemCentroidY_[elem];
        }
    }

    proc updateGhostCellsPhi() {
        // Update ghost cell phi values based on boundary conditions
        inline proc updateWallGhostPhi(face: int) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // For wall boundary, mirror the potential to get zero normal gradient
            this.phi_[ghostElem] = this.phi_[interiorElem];
        }

        inline proc updateFarfieldGhostPhi(face: int) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            this.phi_[ghostElem] = this.inputs_.U_INF_ * this.elemCentroidX_[ghostElem] +
                                  this.inputs_.V_INF_ * this.elemCentroidY_[ghostElem];
        }
        
        forall face in this.mesh_.edgeWall_ do updateWallGhostPhi(face);
        forall face in this.mesh_.edgeFarfield_ do updateFarfieldGhostPhi(face);
    }

    proc updateGhostCellsVelocity() {
        // Update ghost cell velocity values after computing interior velocities
        inline proc updateWallGhostVelocity(face: int) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // Mirror velocity to enforce zero normal velocity at wall
            // V_ghost = V_interior - 2*(V_interior · n)*n
            const nx = this.faceNormalX_[face];
            const ny = this.faceNormalY_[face];
            const uInt = this.uu_[interiorElem];
            const vInt = this.vv_[interiorElem];
            const vDotN = uInt * nx + vInt * ny;
            this.uu_[ghostElem] = uInt - 2.0 * vDotN * nx;
            this.vv_[ghostElem] = vInt - 2.0 * vDotN * ny;
        }

        inline proc updateFarfieldGhostVelocity(face: int) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // Impose U_INF, V_INF at face --> u_face = (u_interior + u_ghost)/2 = U_INF --> u_ghost = 2*U_INF - u_interior
            this.uu_[ghostElem] = 2*this.inputs_.U_INF_ - this.uu_[interiorElem];
            this.vv_[ghostElem] = 2*this.inputs_.V_INF_ - this.vv_[interiorElem];
        }
        
        forall face in this.mesh_.edgeWall_ do updateWallGhostVelocity(face);
        forall face in this.mesh_.edgeFarfield_ do updateFarfieldGhostVelocity(face);
    }

    proc computeGradientGreenGauss(ref phi: [] real(64), 
                                    ref gradX: [] real(64), 
                                    ref gradY: [] real(64)) {
        // Phase 1: Compute and store flux at each face (parallel, no races)
        // Flux direction follows normal: FROM elem1 TO elem2
        forall face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Distance-weighted interpolation to face centroid
            const fcx = this.faceCentroidX_[face];
            const fcy = this.faceCentroidY_[face];
            
            const dx1 = fcx - this.elemCentroidX_[elem1];
            const dy1 = fcy - this.elemCentroidY_[elem1];
            const d1 = sqrt(dx1*dx1 + dy1*dy1);
            
            const dx2 = fcx - this.elemCentroidX_[elem2];
            const dy2 = fcy - this.elemCentroidY_[elem2];
            const d2 = sqrt(dx2*dx2 + dy2*dy2);
            
            // φ_face = (d2 * φ1 + d1 * φ2) / (d1 + d2)
            // This interpolates to the actual face centroid location
            const phiFace = (d2 * phi[elem1] + d1 * phi[elem2]) / (d1 + d2);
            
            // Face flux: φ_face * n * A (positive in normal direction)
            this.faceFluxX_[face] = phiFace * this.faceNormalX_[face] * this.faceArea_[face];
            this.faceFluxY_[face] = phiFace * this.faceNormalY_[face] * this.faceArea_[face];
        }
        
        // Phase 2: Gather fluxes per element with sign correction (parallel)
        // Normal points FROM elem1 TO elem2, so:
        //   - For elem1: flux is outward → add
        //   - For elem2: flux is inward → subtract
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. this.mesh_.elem2edgeIndex_[elem + 1]];
            var gx = 0.0, gy = 0.0;
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                
                // Sign: +1 if we are elem1 (normal points outward), -1 if we are elem2
                const sign = if elem1 == elem then 1.0 else -1.0;
                
                gx += sign * this.faceFluxX_[face];
                gy += sign * this.faceFluxY_[face];
            }
            
            const invVol = 1.0 / this.elemVolume_[elem];
            gradX[elem] = gx * invVol;
            gradY[elem] = gy * invVol;
        }
    }

    proc computeVelocityFromPhi() {
        computeGradientGreenGauss(this.phi_, this.uu_, this.vv_);
    }

    proc computeVelocityFromPhiLeastSquares() {
        this.lsGrad_!.computeGradient(this.phi_, this.uu_, this.vv_, this.elemCentroidX_, this.elemCentroidY_);
    }

    proc computeVelocityFromPhiLeastSquaresQR() {
        this.lsGradQR_!.computeGradient(this.phi_, this.uu_, this.vv_);
    }

    proc computeFluxes() {
        // Compute continuity fluxes using precomputed mesh coefficients.
        // 
        // Face velocity with deferred correction:
        //   V_face = V_avg - (V_avg · t_IJ - dφ/dl) * corrCoeff
        // where:
        //   - V_avg = average of cell-centered velocities
        //   - t_IJ = unit vector from elem1 to elem2 (precomputed)
        //   - dφ/dl = (φ2 - φ1) * invL_IJ (direct phi difference)
        //   - corrCoeff = n / (n · t_IJ) (precomputed)
        
        forall face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];

            // Average cell-centered velocities
            const uAvg = 0.5 * (this.uu_[elem1] + this.uu_[elem2]);
            const vAvg = 0.5 * (this.vv_[elem1] + this.vv_[elem2]);

            // Directional derivative of phi (direct difference)
            const dPhidl = (this.phi_[elem2] - this.phi_[elem1]) * this.invL_IJ_[face];

            // Averaged velocity projected onto cell-to-cell direction
            const vDotT = uAvg * this.t_IJ_x_[face] + vAvg * this.t_IJ_y_[face];

            // Correction: difference between reconstructed and direct gradient
            const delta = vDotT - dPhidl;

            // Apply correction using precomputed coefficients
            const uFace = uAvg - delta * this.corrCoeffX_[face];
            const vFace = vAvg - delta * this.corrCoeffY_[face];
            
            // Isentropic density from Bernoulli equation
            const rhoFace = (1.0 + this.gamma_minus_one_over_two_ * this.inputs_.MACH_ * this.inputs_.MACH_ * 
                             (1.0 - uFace * uFace - vFace * vFace)) ** this.one_over_gamma_minus_one_;

            // Store face quantities
            this.uFace_[face] = uFace;
            this.vFace_[face] = vFace;
            this.rhoFace_[face] = rhoFace;

            // Continuity flux: ρ * (V · n) * A
            const nx = this.faceNormalX_[face];
            const ny = this.faceNormalY_[face];
            // this.flux_[face] = rhoFace * (uFace * nx + vFace * ny) * this.faceArea_[face];
            this.flux_[face] = (uFace * nx + vFace * ny) * this.faceArea_[face];
        }
    }

    proc computeResiduals() {
        // Compute residuals per element from face fluxes
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. this.mesh_.elem2edgeIndex_[elem + 1]];
            var res = 0.0;
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                
                // Sign: +1 if we are elem1 (flux outward), -1 if we are elem2
                const sign = if elem1 == elem then 1.0 else -1.0;
                
                res += sign * this.flux_[face];
            }
            this.res_[elem] = res;
        }
    }

    proc run() {
        this.updateGhostCellsPhi();         // Update ghost phi values for gradient computation
        this.computeVelocityFromPhiLeastSquaresQR();
        this.updateGhostCellsVelocity();    // Update ghost velocities for flux computation
        this.computeFluxes();
        this.computeResiduals();
    }

    proc computeAerodynamicCoefficients() {
        var fx = 0.0;
        var fy = 0.0;
        var Cm = 0.0;
        forall face in this.mesh_.edgeWall_ with (+reduce fx, +reduce fy, +reduce Cm) {
            const rhoFace = this.rhoFace_[face];
            const cpFace = (rhoFace**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            const nx = this.faceNormalX_[face];
            const ny = this.faceNormalY_[face];
            const area = this.faceArea_[face];
            fx += cpFace * nx * area;
            fy += cpFace * ny * area;
            Cm += cpFace * ((this.inputs_.X_REF_ - this.faceCentroidX_[face]) * ny - (this.inputs_.Y_REF_ - this.faceCentroidY_[face]) * nx) * area;
        }

        var Cl = fy*cos(this.inputs_.ALPHA_ * pi / 180.0) - fx*sin(this.inputs_.ALPHA_ * pi / 180.0);
        var Cd = fx*cos(this.inputs_.ALPHA_ * pi / 180.0) + fy*sin(this.inputs_.ALPHA_ * pi / 180.0);

        return (Cl, Cd, Cm);
    }

    proc mach(u: real(64), v: real(64), rho: real(64)): real(64) {
        return this.inputs_.MACH_ * sqrt(u**2 + v**2) * rho**((1-this.inputs_.GAMMA_)/2);
    }

    proc writeSolution(timeList: list(real(64)), 
                       itList: list(int), 
                       resList: list(real(64)), 
                       clList: list(real(64)), 
                       cdList: list(real(64)), 
                       cmList: list(real(64)),
                       circulationList: list(real(64))) {
        const dom = {0..<this.nelemDomain_};
        var phi: [dom] real(64);
        var uu: [dom] real(64);
        var vv: [dom] real(64);
        var ww: [dom] real(64);
        var rhorho: [dom] real(64);
        var pp: [dom] real(64);
        var resres: [dom] real(64);
        var machmach: [dom] real(64);
        var xElem: [dom] real(64);
        var yElem: [dom] real(64);

        forall elem in 1..this.nelemDomain_ {
            phi[elem-1] = this.phi_[elem];
            uu[elem-1] = this.uu_[elem];
            vv[elem-1] = this.vv_[elem];
            rhorho[elem-1] = this.rhorho_[elem];
            pp[elem-1] = (this.rhorho_[elem]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            resres[elem-1] = abs(this.res_[elem]);
            machmach[elem-1] = this.mach(this.uu_[elem], this.vv_[elem], this.rhorho_[elem]);
            xElem[elem-1] = this.elemCentroidX_[elem];
            yElem[elem-1] = this.elemCentroidY_[elem];

        }

        var fields = new map(string, [dom] real(64));
        fields["phi"] = phi;
        fields["VelocityX"] = uu;
        fields["VelocityY"] = vv;
        fields["VelocityZ"] = ww;
        fields["rho"] = rhorho;
        fields["Pressure"] = pp;
        fields["res"] = resres;
        fields["mach"] = machmach;
        fields["xElem"] = xElem;
        fields["yElem"] = yElem;

        var writer = new owned potentialFlowWriter_c(this.inputs_.OUTPUT_FILENAME_);

        // Change X, Y, elem2node, elem2nodeIndex to begin at index 0
        var Xtemp : [0..<this.mesh_.X_.size] real(64);
        var Ytemp : [0..<this.mesh_.Y_.size] real(64);
        for i in 1..this.mesh_.X_.size {
            Xtemp[i-1] = this.mesh_.X_[i];
            Ytemp[i-1] = this.mesh_.Y_[i];
        }
        var elem2nodeTemp : [0..<this.mesh_.elem2node_.size] int;
        for i in 1..this.mesh_.elem2node_.size {
            elem2nodeTemp[i-1] = this.mesh_.elem2node_[i];
        }
        var elem2nodeIndexTemp : [0..<this.mesh_.elem2nodeIndex_.size] int;
        for i in 1..this.mesh_.elem2nodeIndex_.size {
            elem2nodeIndexTemp[i-1] = this.mesh_.elem2nodeIndex_[i];
        }

        writer.writeMeshMultigrid(Xtemp, Ytemp, elem2nodeTemp, elem2nodeIndexTemp);
        writer.writeSolution(dom, fields);

        writer.writeConvergenceHistory(timeList, itList, resList, clList, cdList, cmList, circulationList);

        const wall_dom = {0..<this.mesh_.edgeWall_.size};
        var fieldsWall = new map(string, [wall_dom] real(64));
        var uWall: [wall_dom] real(64);
        var vWall: [wall_dom] real(64);
        var rhoWall: [wall_dom] real(64);
        var pWall: [wall_dom] real(64);
        var machWall: [wall_dom] real(64);
        var xWall: [wall_dom] real(64);
        var yWall: [wall_dom] real(64);
        var nxWall: [wall_dom] real(64);
        var nyWall: [wall_dom] real(64);

        forall (i, face) in zip(wall_dom, this.mesh_.edgeWall_) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // For wall boundary, mirror the velocity
            uWall[i] = this.uFace_[face];
            vWall[i] = this.vFace_[face];
            rhoWall[i] = this.rhoFace_[face];
            pWall[i] = (this.rhoFace_[face]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            machWall[i] = this.mach(this.uu_[interiorElem], this.vv_[interiorElem], this.rhorho_[interiorElem]);
            xWall[i] = this.faceCentroidX_[face];
            yWall[i] = this.faceCentroidY_[face];
            nxWall[i] = this.faceNormalX_[face];
            nyWall[i] = this.faceNormalY_[face];

        }


        fieldsWall["uWall"] = uWall;
        fieldsWall["vWall"] = vWall;
        fieldsWall["rhoWall"] = rhoWall;
        fieldsWall["pressureWall"] = pWall;
        fieldsWall["machWall"] = machWall;
        fieldsWall["xWall"] = xWall;
        fieldsWall["yWall"] = yWall;
        fieldsWall["nxWall"] = nxWall;
        fieldsWall["nyWall"] = nyWall;

        writer.writeWallSolution(this.mesh_, wall_dom, fieldsWall);
    }

    proc writeSolution() {
        const dom = {0..<this.nelemDomain_};
        var phi: [dom] real(64);
        var uu: [dom] real(64);
        var vv: [dom] real(64);
        var ww: [dom] real(64);
        var rhorho: [dom] real(64);
        var pp: [dom] real(64);
        var resres: [dom] real(64);
        var machmach: [dom] real(64);
        var xElem: [dom] real(64);
        var yElem: [dom] real(64);

        forall elem in 1..this.nelemDomain_ {
            phi[elem-1] = this.phi_[elem];
            uu[elem-1] = this.uu_[elem];
            vv[elem-1] = this.vv_[elem];
            rhorho[elem-1] = this.rhorho_[elem];
            pp[elem-1] = (this.rhorho_[elem]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            resres[elem-1] = abs(this.res_[elem]);
            machmach[elem-1] = this.mach(this.uu_[elem], this.vv_[elem], this.rhorho_[elem]);
            xElem[elem-1] = this.elemCentroidX_[elem];
            yElem[elem-1] = this.elemCentroidY_[elem];

        }

        var fields = new map(string, [dom] real(64));
        fields["phi"] = phi;
        fields["VelocityX"] = uu;
        fields["VelocityY"] = vv;
        fields["VelocityZ"] = ww;
        fields["rho"] = rhorho;
        fields["Pressure"] = pp;
        fields["res"] = resres;
        fields["mach"] = machmach;
        fields["xElem"] = xElem;
        fields["yElem"] = yElem;

        var writer = new owned potentialFlowWriter_c(this.inputs_.OUTPUT_FILENAME_);

        // Change X, Y, elem2node, elem2nodeIndex to begin at index 0
        var Xtemp : [0..<this.mesh_.X_.size] real(64);
        var Ytemp : [0..<this.mesh_.Y_.size] real(64);
        for i in 1..this.mesh_.X_.size {
            Xtemp[i-1] = this.mesh_.X_[i];
            Ytemp[i-1] = this.mesh_.Y_[i];
        }
        var elem2nodeTemp : [0..<this.mesh_.elem2node_.size] int;
        for i in 1..this.mesh_.elem2node_.size {
            elem2nodeTemp[i-1] = this.mesh_.elem2node_[i];
        }
        var elem2nodeIndexTemp : [0..<this.mesh_.elem2nodeIndex_.size] int;
        for i in 1..this.mesh_.elem2nodeIndex_.size {
            elem2nodeIndexTemp[i-1] = this.mesh_.elem2nodeIndex_[i];
        }

        writer.writeMeshMultigrid(Xtemp, Ytemp, elem2nodeTemp, elem2nodeIndexTemp);
        writer.writeSolution(dom, fields);

        // writer.writeConvergenceHistory(this.timeList_, this.itList_, this.resList_, this.resPhiList_, this.clList_, this.cdList_, this.cmList_, this.circulationList_);

        const wall_dom = {0..<this.mesh_.edgeWall_.size};
        var fieldsWall = new map(string, [wall_dom] real(64));
        var uWall: [wall_dom] real(64);
        var vWall: [wall_dom] real(64);
        var rhoWall: [wall_dom] real(64);
        var pWall: [wall_dom] real(64);
        var machWall: [wall_dom] real(64);
        var xWall: [wall_dom] real(64);
        var yWall: [wall_dom] real(64);
        var nxWall: [wall_dom] real(64);
        var nyWall: [wall_dom] real(64);

        forall (i, face) in zip(wall_dom, this.mesh_.edgeWall_) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // For wall boundary, mirror the velocity
            uWall[i] = this.uFace_[face];
            vWall[i] = this.vFace_[face];
            rhoWall[i] = this.rhoFace_[face];
            pWall[i] = (this.rhoFace_[face]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            machWall[i] = this.mach(this.uu_[interiorElem], this.vv_[interiorElem], this.rhorho_[interiorElem]);
            xWall[i] = this.faceCentroidX_[face];
            yWall[i] = this.faceCentroidY_[face];
            nxWall[i] = this.faceNormalX_[face];
            nyWall[i] = this.faceNormalY_[face];

        }


        fieldsWall["uWall"] = uWall;
        fieldsWall["vWall"] = vWall;
        fieldsWall["rhoWall"] = rhoWall;
        fieldsWall["pressureWall"] = pWall;
        fieldsWall["machWall"] = machWall;
        fieldsWall["xWall"] = xWall;
        fieldsWall["yWall"] = yWall;
        fieldsWall["nxWall"] = nxWall;
        fieldsWall["nyWall"] = nyWall;

        writer.writeWallSolution(this.mesh_, wall_dom, fieldsWall);
    }
}

}
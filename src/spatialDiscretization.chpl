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

    var faceFluxX_: [face_dom] real(64);   // Flux storage for Green-Gauss
    var faceFluxY_: [face_dom] real(64);
    var flux_: [face_dom] real(64);       // Flux storage for residual computation

    // Least-squares gradient operator (precomputed coefficients)
    var lsGrad_: owned LeastSquaresGradient?;
    var lsGradQR_: owned LeastSquaresGradientQR?;

    proc init(Mesh: shared MeshData, ref inputs: potentialInputs) {
        this.mesh_ = Mesh;
        this.inputs_ = inputs;
        this.nelem_ = this.mesh_.nelemWithGhost_;
        this.nelemDomain_ = this.mesh_.nelem_;
        this.nface_ = this.mesh_.nedge_;

        this.elemDomain_dom = {1..this.nelemDomain_};

        this.elem_dom = {1..this.nelem_};
        
        this.face_dom = {1..this.nface_};
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

        // Initialize and precompute least-squares gradient coefficients
        this.lsGrad_ = new owned LeastSquaresGradient(this.mesh_, this.elemCentroidX_, this.elemCentroidY_);
        this.lsGrad_!.precompute(this.elemCentroidX_, this.elemCentroidY_);

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

    proc updateGhostCells() {
        // Update ghost cell values based on boundary conditions
        inline proc updateWallGhostCell(face: int) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // For wall boundary, mirror the potential
            this.phi_[ghostElem] = this.phi_[interiorElem];
        }

        inline proc updateFarfieldGhostCell(face: int) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // For farfield boundary, set to freestream potential
            const xg = this.elemCentroidX_[ghostElem];
            const yg = this.elemCentroidY_[ghostElem];
            this.phi_[ghostElem] = this.inputs_.U_INF_ * xg + this.inputs_.V_INF_ * yg;
        }
        
        forall face in this.mesh_.edgeWall_ do updateWallGhostCell(face);
        forall face in this.mesh_.edgeFarfield_ do updateFarfieldGhostCell(face);
    }

    /*
     * Debug function to analyze gradient error for a specific element
     */
    proc debugGradientAtElement(elem: int, ref phi: [] real(64)) {
        writeln("=== Debug Gradient for Element ", elem, " ===");
        writeln("Element centroid: (", this.elemCentroidX_[elem], ", ", this.elemCentroidY_[elem], ")");
        writeln("Element volume: ", this.elemVolume_[elem]);
        writeln("phi[elem]: ", phi[elem]);
        
        const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. this.mesh_.elem2edgeIndex_[elem + 1]];
        
        var gx = 0.0, gy = 0.0;
        
        for face in faces {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            const otherElem = if elem1 == elem then elem2 else elem1;
            
            const isBoundary = (otherElem > this.nelemDomain_);
            
            writeln("\n  Face ", face);
            writeln("    elem1=", elem1, ", elem2=", elem2);
            writeln("    Other elem centroid: (", this.elemCentroidX_[otherElem], ", ", this.elemCentroidY_[otherElem], ")");
            writeln("    Face centroid: (", this.faceCentroidX_[face], ", ", this.faceCentroidY_[face], ")");
            
            // Midpoint of the two centroids
            const midX = 0.5 * (this.elemCentroidX_[elem] + this.elemCentroidX_[otherElem]);
            const midY = 0.5 * (this.elemCentroidY_[elem] + this.elemCentroidY_[otherElem]);
            writeln("    Midpoint of centroids: (", midX, ", ", midY, ")");
            
            const phiFace = 0.5 * (phi[elem1] + phi[elem2]);
            writeln("    phi[elem1]=", phi[elem1], ", phi[elem2]=", phi[elem2]);
            writeln("    phi midpoint avg: ", phiFace);
            
            // Distance-weighted interpolation (what the gradient actually uses)
            const dx1 = this.faceCentroidX_[face] - this.elemCentroidX_[elem1];
            const dy1 = this.faceCentroidY_[face] - this.elemCentroidY_[elem1];
            const d1 = sqrt(dx1*dx1 + dy1*dy1);
            const dx2 = this.faceCentroidX_[face] - this.elemCentroidX_[elem2];
            const dy2 = this.faceCentroidY_[face] - this.elemCentroidY_[elem2];
            const d2 = sqrt(dx2*dx2 + dy2*dy2);
            const phiFaceWeighted = (d2 * phi[elem1] + d1 * phi[elem2]) / (d1 + d2);
            writeln("    d1=", d1, ", d2=", d2);
            writeln("    phi weighted: ", phiFaceWeighted);
            
            // What phi SHOULD be at face centroid for linear field
            const phiExactAtFace = this.inputs_.U_INF_ * this.faceCentroidX_[face] + 
                                   this.inputs_.V_INF_ * this.faceCentroidY_[face];
            writeln("    phi exact at face centroid: ", phiExactAtFace);
            writeln("    phi error (midpoint): ", phiFace - phiExactAtFace);
            writeln("    phi error (weighted): ", phiFaceWeighted - phiExactAtFace);
            
            const nx = this.faceNormalX_[face];
            const ny = this.faceNormalY_[face];
            const area = this.faceArea_[face];
            writeln("    normal: (", nx, ", ", ny, "), area: ", area);
            
            const sign = if elem1 == elem then 1.0 else -1.0;
            const fluxX = sign * phiFaceWeighted * nx * area;
            const fluxY = sign * phiFaceWeighted * ny * area;
            writeln("    sign: ", sign, ", fluxX: ", fluxX, ", fluxY: ", fluxY);
            
            gx += fluxX;
            gy += fluxY;
        }
        
        const invVol = 1.0 / this.elemVolume_[elem];
        writeln("\n  Sum of fluxes: (", gx, ", ", gy, ")");
        writeln("  Computed gradient: (", gx * invVol, ", ", gy * invVol, ")");
        writeln("  Expected gradient: (", this.inputs_.U_INF_, ", ", this.inputs_.V_INF_, ")");
        writeln("  Error: (", gx * invVol - this.inputs_.U_INF_, ", ", gy * invVol - this.inputs_.V_INF_, ")");
        writeln("===========================================\n");
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
        // Compute continuity fluxes at each face from averaged states
        forall face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];

            // Average states at face
            const uFace_avg = 0.5 * (this.uu_[elem1] + this.uu_[elem2]);
            const vFace_avg = 0.5 * (this.vv_[elem1] + this.vv_[elem2]);

            // Apply correction
            const l_IJ = sqrt( (this.elemCentroidX_[elem2] - this.elemCentroidX_[elem1])**2 +
                                (this.elemCentroidY_[elem2] - this.elemCentroidY_[elem1])**2 );
            const t_IJ_x = (this.elemCentroidX_[elem2] - this.elemCentroidX_[elem1]) / l_IJ;
            const t_IJ_y = (this.elemCentroidY_[elem2] - this.elemCentroidY_[elem1]) / l_IJ;

            // Directional derivative
            const dPhidl = (this.phi_[elem2] - this.phi_[elem1]) / l_IJ;

            const gradPhiDotT = uFace_avg * t_IJ_x + vFace_avg * t_IJ_y;

            const faceNormalX = this.faceNormalX_[face];
            const faceNormalY = this.faceNormalY_[face];

            const inverseFaceNormalDotT = 1 / (faceNormalX * t_IJ_x + faceNormalY * t_IJ_y);

            const uFace = uFace_avg - (gradPhiDotT - dPhidl) * (faceNormalX * inverseFaceNormalDotT);
            const vFace = vFace_avg - (gradPhiDotT - dPhidl) * (faceNormalY * inverseFaceNormalDotT);
            
            this.uFace_[face] = uFace;
            this.vFace_[face] = vFace;

            // Continuity flux: (u*nx + v*ny) * A
            this.flux_[face] = (uFace * this.faceNormalX_[face] + vFace * this.faceNormalY_[face]) * this.faceArea_[face];
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
        this.updateGhostCells();
        this.computeVelocityFromPhiLeastSquaresQR();
        this.computeFluxes();
        this.computeResiduals();
    }

    proc mach(u: real(64), v: real(64), rho: real(64)): real(64) {
        return this.inputs_.MACH_ * sqrt(u**2 + v**2) * rho**((1-this.inputs_.GAMMA_)/2);
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

        forall (i, face) in zip(wall_dom, this.mesh_.edgeWall_) {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Determine which element is interior and which is ghost
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // For wall boundary, mirror the velocity
            uWall[i] = this.uu_[interiorElem];
            vWall[i] = this.vv_[interiorElem];
            rhoWall[i] = this.rhorho_[interiorElem];
            pWall[i] = (this.rhorho_[interiorElem]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            machWall[i] = this.mach(this.uu_[interiorElem], this.vv_[interiorElem], this.rhorho_[interiorElem]);
            xWall[i] = this.faceCentroidX_[face];
            yWall[i] = this.faceCentroidY_[face];
        }


        fieldsWall["uWall"] = uWall;
        fieldsWall["vWall"] = vWall;
        fieldsWall["rhoWall"] = rhoWall;
        fieldsWall["pressureWall"] = pWall;
        fieldsWall["machWall"] = machWall;
        fieldsWall["xWall"] = xWall;
        fieldsWall["yWall"] = yWall;

        writer.writeWallSolution(this.mesh_, wall_dom, fieldsWall);
    }
}

}
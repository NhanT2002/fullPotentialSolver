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
    var kutta_res_: real(64);

    var elem_dom: domain(1) = {1..0};
    var uu_: [elem_dom] real(64);
    var vv_: [elem_dom] real(64);
    var pp_: [elem_dom] real(64);
    var rhorho_: [elem_dom] real(64);
    var gradRhoX_: [elem_dom] real(64);
    var gradRhoY_: [elem_dom] real(64);
    var phi_: [elem_dom] real(64);
    var elemCentroidX_: [elem_dom] real(64);
    var elemCentroidY_: [elem_dom] real(64);
    var elemVolume_: [elem_dom] real(64);
    var invElemVolume_: [elem_dom] real(64);
    var kuttaCell_: [elem_dom] int; // 1 if over wake, -1 if under wake, 9 otherwise
    var machmach_: [elem_dom] real(64);
    var mumu_: [elem_dom] real(64);

    
    var TEnode_: int;
    var TEnodeXcoord_: real(64);
    var TEnodeYcoord_: real(64);
    var upperTEface_: int;
    var lowerTEface_: int;
    var upperTEelem_: int;
    var lowerTEelem_: int;
    var deltaSupperTEx_: real(64);
    var deltaSupperTEy_: real(64);
    var deltaSlowerTEx_: real(64);
    var deltaSlowerTEy_: real(64);

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
    var machFace_: [face_dom] real(64);
    var velMagFace_: [face_dom] real(64);

    // Precomputed coefficients for flux computation (mesh-dependent only)
    var invL_IJ_: [face_dom] real(64);      // 1 / distance between cell centroids
    var t_IJ_x_: [face_dom] real(64);       // Unit vector from elem1 to elem2 (x-component)
    var t_IJ_y_: [face_dom] real(64);       // Unit vector from elem1 to elem2 (y-component)
    var corrCoeffX_: [face_dom] real(64);   // nx / (n · t_IJ) - correction coefficient (x)
    var corrCoeffY_: [face_dom] real(64);   // ny / (n · t_IJ) - correction coefficient (y)

    var faceFluxX_: [face_dom] real(64);   // Flux storage for Green-Gauss
    var faceFluxY_: [face_dom] real(64);
    var flux_: [face_dom] real(64);       // Flux storage for residual computation
    var upwindElem_: [face_dom] int;       // Upwind element for each face (for Jacobian)
    var downwindElem_: [face_dom] int;     // Downwind element for each face (for Jacobian)

    var gamma_minus_one_over_two_: real(64);
    var one_minus_gamma_over_two_: real(64);
    var one_over_gamma_minus_one_: real(64);

    // Least-squares gradient operator (precomputed coefficients)
    var lsGrad_: owned LeastSquaresGradient?;
    var lsGradQR_: owned LeastSquaresGradientQR?;

    var circulation_ : real(64);

    var wall_dom: sparse subdomain(elemDomain_dom); // cell next to wall boundary
    var fluid_dom: sparse subdomain(elemDomain_dom); // all other cells without wall_dom
    var wake_dom: sparse subdomain(elemDomain_dom); // cells next to wake
    var wallFaceSet_: set(int);  // Set of wall face indices for efficient lookup

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
        this.one_minus_gamma_over_two_ = (1.0 - this.inputs_.GAMMA_) / 2.0;
        this.one_over_gamma_minus_one_ = 1.0 / (this.inputs_.GAMMA_ - 1.0);
    }

    // Update the inputs record (used for Mach continuation)
    proc updateInputs(ref newInputs: potentialInputs) {
        this.inputs_ = newInputs;
    }

    proc initializeMetrics() {
        for face in this.mesh_.edgeWall_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            if elem1 <= this.nelemDomain_ {
                this.wall_dom += elem1;
            }
            else {
                this.wall_dom += elem2;
            }
            this.wallFaceSet_.add(face);  // Add to wall face set for Jacobian
        }
        for elem in 1..this.nelemDomain_ {
            this.fluid_dom += elem;
        }
        this.fluid_dom -= this.wall_dom;

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
            this.invElemVolume_[elem] = 1.0 / vol;
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
        this.lsGradQR_ = new owned LeastSquaresGradientQR(this.mesh_);
        this.lsGradQR_!.precompute(this.elemCentroidX_, this.elemCentroidY_);
    }

    proc initializeKuttaCells() {
        for face in this.mesh_.edgeWall_ {
            for p in this.mesh_.edge2node_[1.., face] {
                const x = this.mesh_.X_[p];
                const y = this.mesh_.Y_[p];

                if x > this.TEnodeXcoord_ {
                    this.TEnode_ = p;
                    this.TEnodeXcoord_ = x;
                    this.TEnodeYcoord_ = y;
                }
            }
        }
        for face in this.mesh_.edgeWall_ {
            const p1 = this.mesh_.edge2node_[1, face];
            const p2 = this.mesh_.edge2node_[2, face];

            if p1 == this.TEnode_ || p2 == this.TEnode_ {
                if this.faceCentroidY_[face] >= 0.0 {
                    this.upperTEface_ = face;
                }
                if this.faceCentroidY_[face] <= 0.0 {
                    this.lowerTEface_ = face;
                }
            }
        }
        this.upperTEelem_ = this.mesh_.edge2elem_[1, this.upperTEface_];
        this.lowerTEelem_ = this.mesh_.edge2elem_[1, this.lowerTEface_];
        writeln("T.E. node: ", this.TEnode_, " at (", this.TEnodeXcoord_, ", ", this.TEnodeYcoord_, ")");
        writeln("Upper T.E. face: ", this.upperTEface_, " elem: ", this.upperTEelem_);
        writeln("Lower T.E. face: ", this.lowerTEface_, " elem: ", this.lowerTEelem_);
        this.kuttaCell_ = 9;
        // Define wake line between (x3, y3) and a far downstream point
        const x3 = this.TEnodeXcoord_;
        const y3 = this.TEnodeYcoord_;
        const x4 = this.TEnodeXcoord_ + 1000.0;
        const y4 = this.TEnodeYcoord_;
        forall elem in 1..this.nelemDomain_ {
            const x = this.elemCentroidX_[elem];
            const y = this.elemCentroidY_[elem];
            if x <= this.TEnodeXcoord_ {
                this.kuttaCell_[elem] = 9;
            }
            else {
                const signedArea = (x4 - x3)*(y - y3) - (y4 - y3)*(x - x3);
                if signedArea >= 0.0 {
                    this.kuttaCell_[elem] = 1;
                }
                else if signedArea < 0.0 {
                    this.kuttaCell_[elem] = -1;
                }

            }
        }
        for face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            if ( (this.kuttaCell_[elem1] == 1 && this.kuttaCell_[elem2] == -1) ||
               (this.kuttaCell_[elem1] == -1 && this.kuttaCell_[elem2] == 1) ) {
                this.wake_dom += elem1;
                this.wake_dom += elem2;
            }
        }

        

        this.deltaSupperTEx_ = this.TEnodeXcoord_ - this.elemCentroidX_[this.upperTEelem_];
        this.deltaSupperTEy_ = this.TEnodeYcoord_ - this.elemCentroidY_[this.upperTEelem_];

        this.deltaSlowerTEx_ = this.TEnodeXcoord_ - this.elemCentroidX_[this.lowerTEelem_];
        this.deltaSlowerTEy_ = this.TEnodeYcoord_ - this.elemCentroidY_[this.lowerTEelem_];
    }

    proc initializeSolution() {
        if this.inputs_.START_FILENAME_ != "" {
            writeln("Initializing solution from file: ", this.inputs_.START_FILENAME_);
            const (xElem, yElem, rho, phi, it, time, res, cl, cd, cm, circulation) = readSolution(this.inputs_.START_FILENAME_);
            if phi.size != this.nelemDomain_ {
                halt("Error: START_FILENAME mesh size does not match current mesh size.");
            }
            else {
                forall elem in 1..this.nelem_ {
                    this.phi_[elem] = phi[elem];
                    this.rhorho_[elem] = rho[elem];
                }
                this.circulation_ = circulation.last;
            }
        }
        else {
            forall elem in 1..this.nelem_ {
                this.uu_[elem] = this.inputs_.U_INF_;
                this.vv_[elem] = this.inputs_.V_INF_;
                this.rhorho_[elem] = this.inputs_.RHO_INF_;
                this.pp_[elem] = this.inputs_.P_INF_;

                this.phi_[elem] = this.inputs_.U_INF_ * this.elemCentroidX_[elem] +
                                this.inputs_.V_INF_ * this.elemCentroidY_[elem];
            }
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

            const x = this.elemCentroidX_[ghostElem];
            const y = this.elemCentroidY_[ghostElem];
            var theta = atan2(y - this.inputs_.Y_REF_, x - this.inputs_.X_REF_);
            if theta < 0.0 {
                theta += 2.0 * pi;
            }
            
            // this.phi_[ghostElem] = this.inputs_.U_INF_ * x + this.inputs_.V_INF_ * y + this.circulation_ * theta / (2.0 * pi);
            
            this.phi_[ghostElem] = this.inputs_.U_INF_ * x + this.inputs_.V_INF_ * y;
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
            const x = this.faceCentroidX_[face];
            const y = this.faceCentroidY_[face];
            // const u_face = this.inputs_.U_INF_ - this.circulation_ / (2.0 * pi) * (y - this.inputs_.Y_REF_) / ((x - this.inputs_.X_REF_)*(x - this.inputs_.X_REF_) + (y - this.inputs_.Y_REF_)*(y - this.inputs_.Y_REF_));
            // const v_face = this.inputs_.V_INF_ + this.circulation_ / (2.0 * pi) * (x - this.inputs_.X_REF_) / ((x - this.inputs_.X_REF_)*(x - this.inputs_.X_REF_) + (y - this.inputs_.Y_REF_)*(y - this.inputs_.Y_REF_));
            const u_face = this.inputs_.U_INF_;
            const v_face = this.inputs_.V_INF_;
            this.uu_[ghostElem] = 2*u_face - this.uu_[interiorElem];
            this.vv_[ghostElem] = 2*v_face - this.vv_[interiorElem];
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
        this.lsGradQR_!.computeGradient(this.phi_, this.uu_, this.vv_, this.kuttaCell_, this.circulation_);
    }

    proc computeDensityFromVelocity() {
        forall elem in 1..this.nelemDomain_ {
            this.rhorho_[elem] = (1.0 + this.gamma_minus_one_over_two_ * this.inputs_.MACH_ * this.inputs_.MACH_ * 
                                 (1.0 - this.uu_[elem] * this.uu_[elem] - this.vv_[elem] * this.vv_[elem])) ** this.one_over_gamma_minus_one_;
            this.machmach_[elem] = this.mach(this.uu_[elem], this.vv_[elem], this.rhorho_[elem]);
            this.mumu_[elem] = this.inputs_.MU_C_ * max(0.0, this.machmach_[elem]*this.machmach_[elem] - this.inputs_.MACH_C_*this.inputs_.MACH_C_);
        }

        forall face in this.mesh_.edgeWall_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            // Determine which element is interior
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);

            // Copy interior density to ghost cell
            this.rhorho_[ghostElem] = this.rhorho_[interiorElem];
            this.machmach_[ghostElem] = this.machmach_[interiorElem];
            this.mumu_[ghostElem] = this.mumu_[interiorElem];
        }

        // Compute gradient of rho
        this.lsGradQR_!.computeGradient(this.rhorho_, this.gradRhoX_, this.gradRhoY_);
    }

    proc computeFaceProperties() {
        // Compute face properties using precomputed mesh coefficients.
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

            // Get phi values with potential jump across wake
            var phi1 = this.phi_[elem1];
            var phi2 = this.phi_[elem2];
            
            // Apply circulation correction for wake-crossing faces
            // From elem1's perspective looking at elem2
            const kuttaType1 = this.kuttaCell_[elem1];
            const kuttaType2 = this.kuttaCell_[elem2];
            if (kuttaType1 == 1 && kuttaType2 == -1) {
                // elem1 is above wake, elem2 is below wake
                // To get continuous potential, add Γ to lower surface value
                phi2 += this.circulation_;
            } else if (kuttaType1 == -1 && kuttaType2 == 1) {
                // elem1 is below wake, elem2 is above wake
                // To get continuous potential, subtract Γ from upper surface value
                phi2 -= this.circulation_;
            }

            // Directional derivative of phi (direct difference with jump correction)
            const dPhidl = (phi2 - phi1) * this.invL_IJ_[face];

            // Averaged velocity projected onto cell-to-cell direction
            const vDotT = uAvg * this.t_IJ_x_[face] + vAvg * this.t_IJ_y_[face];

            // Correction: difference between reconstructed and direct gradient
            const delta = vDotT - dPhidl;

            // Apply correction using precomputed coefficients
            const uFace = uAvg - delta * this.corrCoeffX_[face];
            const vFace = vAvg - delta * this.corrCoeffY_[face];
            
            // Isentropic density from Bernoulli equation
            var rhoFace = (1.0 + this.gamma_minus_one_over_two_ * this.inputs_.MACH_ * this.inputs_.MACH_ * 
                             (1.0 - uFace * uFace - vFace * vFace)) ** this.one_over_gamma_minus_one_;

            // // Padé approximation for density at high Mach numbers
            // const q = sqrt(uFace * uFace + vFace * vFace);
            // if q >= this.inputs_.q_crit_ {
            //     const beta = this.inputs_.beta_crit_;
            //     const rhoCrit = this.inputs_.rho_crit_;
            //     rhoFace = rhoCrit / (1.0 + beta * (q / this.inputs_.q_crit_ - 1.0) );
            // }

            // Store face quantities
            this.uFace_[face] = uFace;
            this.vFace_[face] = vFace;
            this.rhoFace_[face] = rhoFace;
        }
    }

    proc artificialDensity() {
        // Jameson-type density upwinding for transonic stability.
        // 
        // The artificial compressibility formulation blends between isentropic
        // density (accurate for subsonic flow) and upwind density (stable for
        // transonic/supersonic flow):
        //
        //   ρ_face = (1 - ν) * ρ_isentropic + ν * ρ_upwind
        //
        // where ν is a switching function that activates in supersonic regions:
        //   ν = μ_c * max(0, M² - M_c²)
        //
        // This is equivalent to modifying the isentropic density:
        //   ρ_face = ρ_isentropic - ν * (ρ_isentropic - ρ_upwind)
        
        forall face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];

            // Get face velocity and determine upwind direction
            const nx = this.faceNormalX_[face];
            const ny = this.faceNormalY_[face];
            const uFace = this.uFace_[face];
            const vFace = this.vFace_[face];
            const vDotN = uFace * nx + vFace * ny;
            
            // Upwind/downwind elements based on flow direction (store for Jacobian reuse)
            if vDotN >= 0.0 {
                this.upwindElem_[face] = elem1;
                this.downwindElem_[face] = elem2;
            } else {
                this.upwindElem_[face] = elem2;
                this.downwindElem_[face] = elem1;
            }
            const upwindElem = this.upwindElem_[face];
            const downwindElem = this.downwindElem_[face];
            
            // Cell-centered switching function from upwind cell
            const mu = this.mumu_[upwindElem];
            
            // Skip if switching function is zero (subsonic region)
            if mu <= 0.0 then continue;
            
            // Get isentropic and upwind densities
            const rhoIsentropic = this.rhoFace_[face];
            const dx = this.elemCentroidX_[upwindElem] - this.faceCentroidX_[face];
            const dy = this.elemCentroidY_[upwindElem] - this.faceCentroidY_[face];
            const rhoUpwind = this.rhorho_[upwindElem] + 
                                this.gradRhoX_[upwindElem] * dx + 
                                this.gradRhoY_[upwindElem] * dy;
            
            // Blend: ρ_face = ρ_isen - μ * (ρ_isen - ρ_upwind)
            this.rhoFace_[face] = rhoIsentropic - mu * (rhoIsentropic - rhoUpwind);
        }
    }

    proc computeFluxes() {
        // Continuity flux: ρ * (V · n) * A
        forall face in 1..this.nface_ {
            this.flux_[face] = this.rhoFace_[face] * (this.uFace_[face] * this.faceNormalX_[face] 
                            + this.vFace_[face] * this.faceNormalY_[face]) * this.faceArea_[face];

            // Also precompute mach face and velMagFace for jacobian reuse
            this.machFace_[face] = this.mach(this.uFace_[face], this.vFace_[face], this.rhoFace_[face]);
            this.velMagFace_[face] = sqrt(this.uFace_[face]**2 + this.vFace_[face]**2);
        }
    }

    // Lightweight flux computation for Jv - skips machFace and velMagFace
    proc computeFluxes_jv() {
        forall face in 1..this.nface_ {
            this.flux_[face] = this.rhoFace_[face] * (this.uFace_[face] * this.faceNormalX_[face] 
                            + this.vFace_[face] * this.faceNormalY_[face]) * this.faceArea_[face];
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
            this.res_[elem] = res * this.invElemVolume_[elem];
        }

        // Kutta condition residual: R_Γ = Γ - Γ_computed
        // We want Γ to equal the computed value from the potential field
        // R_Γ = Γ_current - (φ_upper - φ_lower) should go to zero
        // Extrapolated phi at upper and lower TE elem
        const phi_upper = this.phi_[this.upperTEelem_] + (this.uu_[this.upperTEelem_] * this.deltaSupperTEx_ + this.vv_[this.upperTEelem_] * this.deltaSupperTEy_);
        const phi_lower = this.phi_[this.lowerTEelem_] + (this.uu_[this.lowerTEelem_] * this.deltaSlowerTEx_ + this.vv_[this.lowerTEelem_] * this.deltaSlowerTEy_);
        const gamma_computed = phi_upper - phi_lower;
        this.kutta_res_ = (this.circulation_ - gamma_computed) / (this.elemVolume_[this.upperTEelem_] + this.elemVolume_[this.lowerTEelem_]);
    }

    proc run() {
        this.updateGhostCellsPhi();         // Update ghost phi values for gradient computation
        this.computeVelocityFromPhiLeastSquaresQR();
        this.computeDensityFromVelocity();
        this.updateGhostCellsVelocity();    // Update ghost velocities for flux computation
        this.computeFaceProperties();
        this.artificialDensity();
        this.computeFluxes();
        this.computeResiduals();
    }

    // Lightweight run for Jacobian-vector products - skips machmach/mumu computation
    // These are only needed for upwinding in artificialDensity, which uses the upwind
    // cell's mumu value. For finite difference Jv, we can reuse the mumu from the
    // base state since the perturbation is small.
    proc run_jv() {
        this.updateGhostCellsPhi();         // Update ghost phi values for gradient computation
        this.computeVelocityFromPhiLeastSquaresQR();
        this.computeDensityFromVelocity();
        this.updateGhostCellsVelocity();    // Update ghost velocities for flux computation
        this.computeFaceProperties();
        this.artificialDensity();
        this.computeFluxes();
        this.computeResiduals();
    }

    // Lightweight density computation - skips machmach and mumu (reuse from base state)
    proc computeDensityFromVelocity_jv() {
        forall elem in 1..this.nelemDomain_ {
            this.rhorho_[elem] = (1.0 + this.gamma_minus_one_over_two_ * this.inputs_.MACH_ * this.inputs_.MACH_ * 
                                 (1.0 - this.uu_[elem] * this.uu_[elem] - this.vv_[elem] * this.vv_[elem])) ** this.one_over_gamma_minus_one_;
            // Skip machmach and mumu - they're only needed for upwinding direction
            // and we reuse the values from the unperturbed state
        }

        forall face in this.mesh_.edgeWall_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            this.rhorho_[ghostElem] = this.rhorho_[interiorElem];
        }
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
        return this.inputs_.MACH_ * sqrt(u**2 + v**2) * rho**this.one_minus_gamma_over_two_;
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
        var gradRhoX: [dom] real(64);
        var gradRhoY: [dom] real(64);
        var pp: [dom] real(64);
        var resres: [dom] real(64);
        var machmach: [dom] real(64);
        var xElem: [dom] real(64);
        var yElem: [dom] real(64);
        var kuttaCell: [dom] int;

        forall elem in 1..this.nelemDomain_ {
            phi[elem-1] = this.phi_[elem];
            uu[elem-1] = this.uu_[elem];
            vv[elem-1] = this.vv_[elem];
            rhorho[elem-1] = this.rhorho_[elem];
            gradRhoX[elem-1] = this.gradRhoX_[elem];
            gradRhoY[elem-1] = this.gradRhoY_[elem];
            pp[elem-1] = (this.rhorho_[elem]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            resres[elem-1] = abs(this.res_[elem] * this.elemVolume_[elem]);
            machmach[elem-1] = this.mach(this.uu_[elem], this.vv_[elem], this.rhorho_[elem]);
            xElem[elem-1] = this.elemCentroidX_[elem];
            yElem[elem-1] = this.elemCentroidY_[elem];
            kuttaCell[elem-1] = this.kuttaCell_[elem];
        }

        var fields = new map(string, [dom] real(64));
        fields["phi"] = phi;
        fields["VelocityX"] = uu;
        fields["VelocityY"] = vv;
        fields["VelocityZ"] = ww;
        fields["rho"] = rhorho;
        fields["gradRhoX"] = gradRhoX;
        fields["gradRhoY"] = gradRhoY;
        fields["cp"] = pp;
        fields["res"] = resres;
        fields["mach"] = machmach;
        fields["xElem"] = xElem;
        fields["yElem"] = yElem;
        fields["kuttaCell"] = kuttaCell;

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
        fieldsWall["cpWall"] = pWall;
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
        var gradRhoX: [dom] real(64);
        var gradRhoY: [dom] real(64);
        var pp: [dom] real(64);
        var resres: [dom] real(64);
        var machmach: [dom] real(64);
        var xElem: [dom] real(64);
        var yElem: [dom] real(64);
        var kuttaCell: [dom] int;

        forall elem in 1..this.nelemDomain_ {
            phi[elem-1] = this.phi_[elem];
            uu[elem-1] = this.uu_[elem];
            vv[elem-1] = this.vv_[elem];
            rhorho[elem-1] = this.rhorho_[elem];
            gradRhoX[elem-1] = this.gradRhoX_[elem];
            gradRhoY[elem-1] = this.gradRhoY_[elem];
            pp[elem-1] = (this.rhorho_[elem]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            resres[elem-1] = abs(this.res_[elem] * this.elemVolume_[elem]);
            machmach[elem-1] = this.mach(this.uu_[elem], this.vv_[elem], this.rhorho_[elem]);
            xElem[elem-1] = this.elemCentroidX_[elem];
            yElem[elem-1] = this.elemCentroidY_[elem];
            kuttaCell[elem-1] = this.kuttaCell_[elem];
        }

        var fields = new map(string, [dom] real(64));
        fields["phi"] = phi;
        fields["VelocityX"] = uu;
        fields["VelocityY"] = vv;
        fields["VelocityZ"] = ww;
        fields["rho"] = rhorho;
        fields["gradRhoX"] = gradRhoX;
        fields["gradRhoY"] = gradRhoY;
        fields["cp"] = pp;
        fields["res"] = resres;
        fields["mach"] = machmach;
        fields["xElem"] = xElem;
        fields["yElem"] = yElem;
        fields["kuttaCell"] = kuttaCell;

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
        fieldsWall["cpWall"] = pWall;
        fieldsWall["machWall"] = machWall;
        fieldsWall["xWall"] = xWall;
        fieldsWall["yWall"] = yWall;
        fieldsWall["nxWall"] = nxWall;
        fieldsWall["nyWall"] = nyWall;

        writer.writeWallSolution(this.mesh_, wall_dom, fieldsWall);
    }
}

}
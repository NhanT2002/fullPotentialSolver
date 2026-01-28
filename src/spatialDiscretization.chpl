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

// config const elemID : int = 0;   // Element index for debugging purposes

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
    var graduuX_: [elem_dom] real(64);
    var graduuY_: [elem_dom] real(64);
    var vv_: [elem_dom] real(64);
    var gradvvX_: [elem_dom] real(64);
    var gradvvY_: [elem_dom] real(64);
    var pp_: [elem_dom] real(64);
    var rhorho_: [elem_dom] real(64);
    var gradRhoX_: [elem_dom] real(64);
    var gradRhoY_: [elem_dom] real(64);
    var phi_: [elem_dom] real(64);
    var elemCentroidX_: [elem_dom] real(64);
    var elemCentroidY_: [elem_dom] real(64);
    var elemVolume_: [elem_dom] real(64);
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
    var res_scale_: real(64);

    var face_dom: domain(1) = {1..0};
    var faceCentroidX_: [face_dom] real(64);
    var faceCentroidY_: [face_dom] real(64);
    var faceArea_: [face_dom] real(64);
    var faceNormalX_: [face_dom] real(64);
    var faceNormalY_: [face_dom] real(64);
    var weights1_: [face_dom] real(64);
    var weights2_: [face_dom] real(64);
    var uFace_: [face_dom] real(64);
    var vFace_: [face_dom] real(64);
    var rhoFace_: [face_dom] real(64);
    var rhoIsenFace_: [face_dom] real(64);
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
    var wake_face_dom: domain(1) = {1..0};
    var wakeFace_: [wake_face_dom] int;
    var wakeFaceUpper_: [wake_face_dom] int;
    var wakeFaceLower_: [wake_face_dom] int;
    var wakeFace2index_: map(int, int);
    var wakeFaceX_: [wake_face_dom] real(64);
    var wakeFaceY_: [wake_face_dom] real(64);
    var wakeFaceZ_: [wake_face_dom] real(64);
    var wakeFaceGamma_: [wake_face_dom] real(64);
    var wakeFaceIndexInfluenceOnElem_: [elem_dom] int;

    var wall_dom: sparse subdomain(elemDomain_dom); // cell next to wall boundary
    var farfield_dom: sparse subdomain(elemDomain_dom); // cell next to farfield boundary
    var fluid_dom: sparse subdomain(elemDomain_dom); // all other cells without wall_dom
    var wake_dom: sparse subdomain(elemDomain_dom); // cells next to wake
    var shock_dom: sparse subdomain(elemDomain_dom); // cells next to shock
    var wallFaceSet_: set(int);  // Set of wall face indices for efficient lookup

    var FREEZE_MU_: bool = false;

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
        for face in this.mesh_.edgeFarfield_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            if elem1 <= this.nelemDomain_ {
                this.farfield_dom += elem1;
            }
            else {
                this.farfield_dom += elem2;
            }
        }
        for elem in 1..this.nelemDomain_ {
            this.fluid_dom += elem;
        }
        this.fluid_dom -= this.wall_dom;
        this.fluid_dom -= this.farfield_dom;

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

        const min_volume = min reduce this.elemVolume_[1..this.nelemDomain_];
        this.res_scale_ = 1.0 / 1; // Used for scaling residuals

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

        // compute weigts for flux average gradient
        forall face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];

            const faceCx = this.faceCentroidX_[face];
            const faceCy = this.faceCentroidY_[face];
            const elem1Cx = this.elemCentroidX_[elem1];
            const elem1Cy = this.elemCentroidY_[elem1];
            const elem2Cx = this.elemCentroidX_[elem2];
            const elem2Cy = this.elemCentroidY_[elem2];

            const distElem1X = faceCx - elem1Cx;
            const distElem1Y = faceCy - elem1Cy;
            const distElem2X = faceCx - elem2Cx;
            const distElem2Y = faceCy - elem2Cy;
            const distElem1 = sqrt(distElem1X * distElem1X + distElem1Y * distElem1Y);
            const distElem2 = sqrt(distElem2X * distElem2X + distElem2Y * distElem2Y);

            const weight1 = distElem2 / (distElem1 + distElem2);
            const weight2 = distElem1 / (distElem1 + distElem2);

            this.weights1_[face] = weight1;
            this.weights2_[face] = weight2;
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
        this.kuttaCell_ = 0;
        // Define wake line between (x3, y3) and a far downstream point
        const x3 = this.TEnodeXcoord_;
        const y3 = this.TEnodeYcoord_;
        const x4 = this.TEnodeXcoord_ + 1000.0;
        const y4 = this.TEnodeYcoord_;
        forall elem in 1..this.nelemDomain_ {
            const x = this.elemCentroidX_[elem];
            const y = this.elemCentroidY_[elem];
            if x <= this.TEnodeXcoord_ {
                this.kuttaCell_[elem] = 0;
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
        var wake_face_list = new list((real(64), int));
        for face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            if ( (this.kuttaCell_[elem1] == 1 && this.kuttaCell_[elem2] == -1) ||
               (this.kuttaCell_[elem1] == -1 && this.kuttaCell_[elem2] == 1) ) {
                this.wake_dom += elem1;
                this.wake_dom += elem2;
                wake_face_list.pushBack((this.faceCentroidX_[face], face));
            }
        }
        this.kuttaCell_ = 0;
        sort(wake_face_list);
        this.wake_face_dom = {1..wake_face_list.size};
        forall i in this.wake_face_dom {
            this.wakeFace_[i] = wake_face_list[i-1][1];
            this.wakeFaceX_[i] = this.faceCentroidX_[this.wakeFace_[i]];
            this.wakeFaceY_[i] = this.faceCentroidY_[this.wakeFace_[i]];

            const elem1 = this.mesh_.edge2elem_[1, this.wakeFace_[i]];
            const elem2 = this.mesh_.edge2elem_[2, this.wakeFace_[i]];
            if this.elemCentroidY_[elem1] > this.elemCentroidY_[elem2] {
                this.kuttaCell_[elem1] = 1;
                this.kuttaCell_[elem2] = -1;
                this.wakeFaceUpper_[i] = elem1;
                this.wakeFaceLower_[i] = elem2;
            }
            else {
                this.kuttaCell_[elem1] = -1;
                this.kuttaCell_[elem2] = 1;
                this.wakeFaceUpper_[i] = elem2;
                this.wakeFaceLower_[i] = elem1;
            }
        }
        for i in this.wake_face_dom {
            this.wakeFace2index_[this.wakeFace_[i]] = i;
        }

        this.wakeFaceIndexInfluenceOnElem_ = -1;
        for (i, face) in zip(this.wakeFace_.domain, this.wakeFace_) {
            if i == 1 {
                const elemUpper = this.wakeFaceUpper_[i];
                const elemLower = this.wakeFaceLower_[i];
                this.wakeFaceIndexInfluenceOnElem_[elemUpper] = i;
                this.wakeFaceIndexInfluenceOnElem_[elemLower] = i;

                const neighborFacesUpper = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elemUpper]+1 .. this.mesh_.elem2edgeIndex_[elemUpper+1]];
                for faceNeighbor in neighborFacesUpper {
                    const elem1 = this.mesh_.edge2elem_[1, faceNeighbor];
                    const elem2 = this.mesh_.edge2elem_[2, faceNeighbor];
                    const neighborElem = if elem1 == elemUpper then elem2 else elem1;
                    if this.kuttaCell_[neighborElem] == 0 {
                        this.wakeFaceIndexInfluenceOnElem_[neighborElem] = i;
                    }
                }

                const neighborFacesLower = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elemLower]+1 .. this.mesh_.elem2edgeIndex_[elemLower+1]];
                for faceNeighbor in neighborFacesLower {
                    const elem1 = this.mesh_.edge2elem_[1, faceNeighbor];
                    const elem2 = this.mesh_.edge2elem_[2, faceNeighbor];
                    const neighborElem = if elem1 == elemLower then elem2 else elem1;
                    if this.kuttaCell_[neighborElem] == 0 {
                        this.wakeFaceIndexInfluenceOnElem_[neighborElem] = i;
                    }
                }
            }
            else {
                const elemUpper = this.wakeFaceUpper_[i];
                const elemLower = this.wakeFaceLower_[i];
                this.wakeFaceIndexInfluenceOnElem_[elemUpper] = i-1;
                this.wakeFaceIndexInfluenceOnElem_[elemLower] = i-1;

                const neighborFacesUpper = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elemUpper]+1 .. this.mesh_.elem2edgeIndex_[elemUpper+1]];
                for faceNeighbor in neighborFacesUpper {
                    const elem1 = this.mesh_.edge2elem_[1, faceNeighbor];
                    const elem2 = this.mesh_.edge2elem_[2, faceNeighbor];
                    const neighborElem = if elem1 == elemUpper then elem2 else elem1;
                    if this.kuttaCell_[neighborElem] == 0 {
                        this.wakeFaceIndexInfluenceOnElem_[neighborElem] = i-1;
                    }
                }

                const neighborFacesLower = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elemLower]+1 .. this.mesh_.elem2edgeIndex_[elemLower+1]];
                for faceNeighbor in neighborFacesLower {
                    const elem1 = this.mesh_.edge2elem_[1, faceNeighbor];
                    const elem2 = this.mesh_.edge2elem_[2, faceNeighbor];
                    const neighborElem = if elem1 == elemLower then elem2 else elem1;
                    if this.kuttaCell_[neighborElem] == 0 {
                        this.wakeFaceIndexInfluenceOnElem_[neighborElem] = i-1;
                    }
                }
            }
        }

        // for elem in 1..this.nelemDomain_ {
        //     if this.wakeFaceIndexInfluenceOnElem_[elem] != -1 {
        //         const faceIndex = this.wakeFaceIndexInfluenceOnElem_[elem];
        //         const upperElem = this.wakeFaceUpper_[faceIndex];
        //         const lowerElem = this.wakeFaceLower_[faceIndex];
        //         writeln("Elem ", elem, " influenced by wake face ", faceIndex, 
        //                 " between elems ", upperElem, " and ", lowerElem);
        //     }
        // }
        

        this.deltaSupperTEx_ = this.TEnodeXcoord_ - this.elemCentroidX_[this.upperTEelem_];
        this.deltaSupperTEy_ = this.TEnodeYcoord_ - this.elemCentroidY_[this.upperTEelem_];

        this.deltaSlowerTEx_ = this.TEnodeXcoord_ - this.elemCentroidX_[this.lowerTEelem_];
        this.deltaSlowerTEy_ = this.TEnodeYcoord_ - this.elemCentroidY_[this.lowerTEelem_];
    }

    proc initializeSolution() {
        if this.inputs_.START_FILENAME_ != "" {
            writeln("Initializing solution from file: ", this.inputs_.START_FILENAME_);
            const (xElem, yElem, rho, phi, it, time, res, cl, cd, cm, circulation, wakeGamma) = readSolution(this.inputs_.START_FILENAME_);
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

        this.updateGhostCellsPhi();
        this.computeVelocityFromPhiLeastSquaresQR();
        this.computeDensityFromVelocity();
        this.updateGhostCellsVelocity();
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
            
            if this.inputs_.FARFIELD_BC_TYPE_ == "cylinder" {
                // Analytical cylinder solution: φ = U_∞ * (r + R²/r) * cos(θ)
                // where R is the cylinder radius
                const R = this.inputs_.CYLINDER_RADIUS_;
                const r2 = x*x + y*y;
                const r = sqrt(r2);
                const theta = atan2(y, x);
                this.phi_[ghostElem] = this.inputs_.VEL_INF_ * (r + R*R/r) * cos(theta);
            }
            else {
                // Default freestream BC
                this.phi_[ghostElem] = this.inputs_.U_INF_ * x + this.inputs_.V_INF_ * y;
            }
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
            
            // Impose velocity at face --> u_face = (u_interior + u_ghost)/2 --> u_ghost = 2*u_face - u_interior
            const x = this.faceCentroidX_[face];
            const y = this.faceCentroidY_[face];
            
            var u_face: real(64);
            var v_face: real(64);
            
            if this.inputs_.FARFIELD_BC_TYPE_ == "cylinder" {
                // Analytical cylinder velocity:
                // V_r = U_∞ * (1 - R²/r²) * cos(θ)
                // V_θ = -U_∞ * (1 + R²/r²) * sin(θ)
                // u = V_r * cos(θ) - V_θ * sin(θ)
                // v = V_r * sin(θ) + V_θ * cos(θ)
                const R = this.inputs_.CYLINDER_RADIUS_;
                const r2 = x*x + y*y;
                const R2_over_r2 = R*R / r2;
                const theta = atan2(y, x);
                const cos_theta = cos(theta);
                const sin_theta = sin(theta);
                
                const Vr = this.inputs_.VEL_INF_ * (1.0 - R2_over_r2) * cos_theta;
                const Vtheta = -this.inputs_.VEL_INF_ * (1.0 + R2_over_r2) * sin_theta;
                
                u_face = Vr * cos_theta - Vtheta * sin_theta;
                v_face = Vr * sin_theta + Vtheta * cos_theta;
            }
            else {
                // Default freestream BC
                u_face = this.inputs_.U_INF_;
                v_face = this.inputs_.V_INF_;
            }
            
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
            
            if this.FREEZE_MU_ == false {
                // Linear switching function:
                // μ = μ_C * max(0, M² - M_C²)
                // Properties: μ = 0 for M ≤ M_C, μ grows unboundedly with M² (sharper shock)
                const M2 = this.machmach_[elem] * this.machmach_[elem];
                const Mc2 = this.inputs_.MACH_C_ * this.inputs_.MACH_C_;
                this.mumu_[elem] = this.inputs_.MU_C_ * max(0.0, M2 - Mc2);
            }
        }

        // forall elem in 1..this.nelemDomain_ {
        //     const neighbors = this.mesh_.esuel_[this.mesh_.esuelIndex_[elem] + 1 .. this.mesh_.esuelIndex_[elem + 1]];
        //     var maxMu = this.mumu_[elem];
        //     for neighbor in neighbors {
        //         maxMu = max(maxMu, this.mumu_[neighbor]);
        //     }
        //     this.mumu_[elem] = maxMu;

        //     // var avgMu = this.mumu_[elem];
        //     // for neighbor in neighbors {
        //     //     avgMu += this.mumu_[neighbor];
        //     // }
        //     // avgMu /= (1 + neighbors.size);
        //     // this.mumu_[elem] = avgMu;
        // }

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
        // Compute shock_dom based on density gradient magnitude
        this.shock_dom.clear();
        for face in this.mesh_.edgeWall_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            // Determine which element is interior
            const (interiorElem, ghostElem) = 
                if elem1 <= this.nelemDomain_ then (elem1, elem2) else (elem2, elem1);

            const gradRhoMag = sqrt(this.gradRhoX_[interiorElem]*this.gradRhoX_[interiorElem] + 
                                    this.gradRhoY_[interiorElem]*this.gradRhoY_[interiorElem]);
            const machCell = this.machmach_[interiorElem];
            if gradRhoMag >= 1.0 && machCell >= 0.9 && this.elemCentroidX_[interiorElem] >= 0.5 {
                this.shock_dom += interiorElem;
            }
        }
    }

    proc computeFaceProperties() {
        // Compute face properties using precomputed mesh coefficients.
        // Face velocity with deferred correction:
        //   V_face = V_avg - (V_avg · t_IJ - dφ/dl) * corrCoeff
        // where:
        //   - V_avg = reconstructed average of cell velocities at face
        //   - t_IJ = unit vector from elem1 to elem2 (precomputed)
        //   - dφ/dl = (φ2 - φ1) * invL_IJ (direct phi difference)
        //   - corrCoeff = n / (n · t_IJ) (precomputed)
        
        for face in 1..this.nface_ {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];

            const weight1 = this.weights1_[face];
            const weight2 = this.weights2_[face];
            
            const uAvg = weight1 * this.uu_[elem1] + weight2 * this.uu_[elem2];
            const vAvg = weight1 * this.vv_[elem1] + weight2 * this.vv_[elem2];

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
            this.rhoIsenFace_[face] = rhoFace; // Store isentropic density for later use
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
            // Simplified: no gradient extrapolation for Jacobian consistency
            // this.rhoFace_[face] = this.rhoNonIsentropic(this.rhoFace_[face], this.machmach_[upwindElem]);
            const rhoIsentropic = this.rhoFace_[face];
            const rhoUpwind = this.rhorho_[upwindElem];
            
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
            this.res_[elem] = res * this.res_scale_;
        }

        if this.inputs_.FREEZE_CIRCULATION_ == false {
            // Kutta condition residual: R_Γ = Γ - Γ_computed
            // We want Γ to equal the computed value from the potential field
            // R_Γ = Γ_current - (φ_upper - φ_lower) should go to zero
            // Extrapolated phi at upper and lower TE elem
            const phi_upper = this.phi_[this.upperTEelem_] + (this.uu_[this.upperTEelem_] * this.deltaSupperTEx_ + this.vv_[this.upperTEelem_] * this.deltaSupperTEy_);
            const phi_lower = this.phi_[this.lowerTEelem_] + (this.uu_[this.lowerTEelem_] * this.deltaSlowerTEx_ + this.vv_[this.lowerTEelem_] * this.deltaSlowerTEy_);
            const gamma_computed = phi_upper - phi_lower;
            this.kutta_res_ = (this.circulation_ - gamma_computed) * this.res_scale_;
        }
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
            const rhoFace = this.rhoIsenFace_[face];
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

    proc rhoNonIsentropic(rhoIsen : real(64), machUpstream : real(64)) {
        const ds_R = 1 / (this.inputs_.GAMMA_ - 1.0) * ln((2*this.inputs_.GAMMA_ * machUpstream**2 - (this.inputs_.GAMMA_ - 1.0)) / (this.inputs_.GAMMA_ + 1.0))
            - this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ - 1.0) * ln((this.inputs_.GAMMA_ + 1.0) * machUpstream**2 / ((this.inputs_.GAMMA_ - 1.0) * machUpstream**2 + 2.0));
        return rhoIsen * exp(-ds_R);
    }

    proc cylinder_solution(x: [] real(64), y: [] real(64)) {
        const dom = x.domain;
        var phi_exact: [dom] real(64);
        var Vr_exact: [dom] real(64);
        var Vtheta_exact: [dom] real(64);
        var u_exact: [dom] real(64);
        var v_exact: [dom] real(64);
        const R = this.inputs_.CYLINDER_RADIUS_;
        forall i in x.domain {
            const r = sqrt(x[i]**2 + y[i]**2);
            const theta = atan2(y[i], x[i]);
            phi_exact[i] = this.inputs_.VEL_INF_ * (r + R**2 / r) * cos(theta);
            Vr_exact[i] = this.inputs_.VEL_INF_ * (1.0 - R**2 / r**2) * cos(theta);
            Vtheta_exact[i] = -this.inputs_.VEL_INF_ * (1.0 + R**2 / r**2) * sin(theta);
            u_exact[i] = Vr_exact[i] * cos(theta) - Vtheta_exact[i] * sin(theta);
            v_exact[i] = Vr_exact[i] * sin(theta) + Vtheta_exact[i] * cos(theta);
        }
        return (phi_exact, u_exact, v_exact);
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
        var graduX: [dom] real(64);
        var graduY: [dom] real(64);
        var vv: [dom] real(64);
        var gradvX: [dom] real(64);
        var gradvY: [dom] real(64);
        var ww: [dom] real(64);
        var rhorho: [dom] real(64);
        var gradRhoX: [dom] real(64);
        var gradRhoY: [dom] real(64);
        var pp: [dom] real(64);
        var resres: [dom] real(64);
        var machmach: [dom] real(64);
        var xElem: [dom] real(64);
        var yElem: [dom] real(64);
        var volumeElem: [dom] real(64);
        var kuttaCell: [dom] int;

        forall elem in 1..this.nelemDomain_ {
            phi[elem-1] = this.phi_[elem];
            uu[elem-1] = this.uu_[elem];
            graduX[elem-1] = this.graduuX_[elem];
            graduY[elem-1] = this.graduuY_[elem];
            vv[elem-1] = this.vv_[elem];
            gradvX[elem-1] = this.gradvvX_[elem];
            gradvY[elem-1] = this.gradvvY_[elem];
            rhorho[elem-1] = this.rhorho_[elem];
            gradRhoX[elem-1] = this.gradRhoX_[elem];
            gradRhoY[elem-1] = this.gradRhoY_[elem];
            pp[elem-1] = (this.rhorho_[elem]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            resres[elem-1] = abs(this.res_[elem] / this.res_scale_);
            machmach[elem-1] = this.mach(this.uu_[elem], this.vv_[elem], this.rhorho_[elem]);
            xElem[elem-1] = this.elemCentroidX_[elem];
            yElem[elem-1] = this.elemCentroidY_[elem];
            volumeElem[elem-1] = this.elemVolume_[elem];
            kuttaCell[elem-1] = this.kuttaCell_[elem];
        }

        var fields = new map(string, [dom] real(64));
        fields["phi"] = phi;
        fields["VelocityX"] = uu;
        fields["graduX"] = graduX;
        fields["graduY"] = graduY;
        fields["VelocityY"] = vv;
        fields["gradvX"] = gradvX;
        fields["gradvY"] = gradvY;
        fields["VelocityZ"] = ww;
        fields["rho"] = rhorho;
        fields["gradRhoX"] = gradRhoX;
        fields["gradRhoY"] = gradRhoY;
        fields["cp"] = pp;
        fields["res"] = resres;
        fields["mach"] = machmach;
        fields["xElem"] = xElem;
        fields["yElem"] = yElem;
        fields["volumeElem"] = volumeElem;
        fields["kuttaCell"] = kuttaCell;

        if this.inputs_.FARFIELD_BC_TYPE_ == "cylinder" {
            const (phi_exact, u_exact, v_exact) = this.cylinder_solution(xElem, yElem);
            var error_phi : [dom] real(64);
            var error_u : [dom] real(64);
            var error_v : [dom] real(64);
            forall i in dom {
                error_phi[i] = abs(phi[i] - phi_exact[i]);
                error_u[i] = abs(uu[i] - u_exact[i]);
                error_v[i] = abs(vv[i] - v_exact[i]);
            }

            fields["phi_exact"] = phi_exact;
            fields["u_exact"] = u_exact;
            fields["v_exact"] = v_exact;
            fields["error_phi"] = error_phi;
            fields["error_u"] = error_u;
            fields["error_v"] = error_v;

            const rmse_error_u = RMSE(error_u);
            const rmse_error_v = RMSE(error_v);

            writeln("RMSE Error in u: ", rmse_error_u, ", v: ", rmse_error_v);
        }
        
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
            machWall[i] = this.mach(this.uFace_[face], this.vFace_[face], this.rhoFace_[face]);
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

        if this.inputs_.FARFIELD_BC_TYPE_ == "cylinder" {
            const (phi_exact, u_exact, v_exact) = this.cylinder_solution(xWall, yWall);
            var error_u_wall : [wall_dom] real(64);
            var error_v_wall : [wall_dom] real(64);
            forall i in wall_dom {
                error_u_wall[i] = abs(uWall[i] - u_exact[i]);
                error_v_wall[i] = abs(vWall[i] - v_exact[i]);
            }

            const rmse_error_u_wall = RMSE(error_u_wall);
            const rmse_error_v_wall = RMSE(error_v_wall);

            writeln("RMSE Error wall in u: ", rmse_error_u_wall, ", v: ", rmse_error_v_wall);
        }

        writer.writeWallSolution(this.mesh_, wall_dom, fieldsWall);

        const wake_dom = {0..<this.wake_face_dom.size};
        var fieldsWake = new map(string, [wake_dom] real(64));
        var uWake: [wake_dom] real(64);
        var vWake: [wake_dom] real(64);
        var rhoWake: [wake_dom] real(64);
        var pWake: [wake_dom] real(64);
        var machWake: [wake_dom] real(64);
        var xWake: [wake_dom] real(64);
        var yWake: [wake_dom] real(64);
        var nxWake: [wake_dom] real(64);
        var nyWake: [wake_dom] real(64);
        var gammaWake: [wake_dom] real(64);

        forall (i, face) in zip(this.wake_face_dom, this.wakeFace_) {
            uWake[i-1] = this.uFace_[face];
            vWake[i-1] = this.vFace_[face];
            rhoWake[i-1] = this.rhoFace_[face];
            pWake[i-1] = (this.rhoFace_[face]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            machWake[i-1] = this.machFace_[face];
            xWake[i-1] = this.faceCentroidX_[face];
            yWake[i-1] = this.faceCentroidY_[face];
            nxWake[i-1] = this.faceNormalX_[face];
            nyWake[i-1] = this.faceNormalY_[face];
            gammaWake[i-1] = this.circulation_;
        }

        fieldsWake["uWake"] = uWake;
        fieldsWake["vWake"] = vWake;
        fieldsWake["rhoWake"] = rhoWake;
        fieldsWake["cpWake"] = pWake;
        fieldsWake["machWake"] = machWake;
        fieldsWake["xWake"] = xWake;
        fieldsWake["yWake"] = yWake;
        fieldsWake["nxWake"] = nxWake;
        fieldsWake["nyWake"] = nyWake;
        fieldsWake["gammaWake"] = gammaWake;

        writer.writeWakeToCGNS(this.wakeFaceX_, this.wakeFaceY_, this.wakeFaceZ_, wake_dom, fieldsWake);
    }

    proc writeSolution() {
        const dom = {0..<this.nelemDomain_};
        var phi: [dom] real(64);
        var uu: [dom] real(64);
        var graduX: [dom] real(64);
        var graduY: [dom] real(64);
        var vv: [dom] real(64);
        var gradvX: [dom] real(64);
        var gradvY: [dom] real(64);
        var ww: [dom] real(64);
        var rhorho: [dom] real(64);
        var gradRhoX: [dom] real(64);
        var gradRhoY: [dom] real(64);
        var pp: [dom] real(64);
        var resres: [dom] real(64);
        var machmach: [dom] real(64);
        var xElem: [dom] real(64);
        var yElem: [dom] real(64);
        var volumeElem: [dom] real(64);
        var kuttaCell: [dom] int;

        forall elem in 1..this.nelemDomain_ {
            phi[elem-1] = this.phi_[elem];
            uu[elem-1] = this.uu_[elem];
            graduX[elem-1] = this.graduuX_[elem];
            graduY[elem-1] = this.graduuY_[elem];
            vv[elem-1] = this.vv_[elem];
            gradvX[elem-1] = this.gradvvX_[elem];
            gradvY[elem-1] = this.gradvvY_[elem];
            rhorho[elem-1] = this.rhorho_[elem];
            gradRhoX[elem-1] = this.gradRhoX_[elem];
            gradRhoY[elem-1] = this.gradRhoY_[elem];
            pp[elem-1] = (this.rhorho_[elem]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            resres[elem-1] = abs(this.res_[elem] / this.res_scale_);
            machmach[elem-1] = this.mach(this.uu_[elem], this.vv_[elem], this.rhorho_[elem]);
            xElem[elem-1] = this.elemCentroidX_[elem];
            yElem[elem-1] = this.elemCentroidY_[elem];
            volumeElem[elem-1] = this.elemVolume_[elem];
            kuttaCell[elem-1] = this.kuttaCell_[elem];
        }

        var fields = new map(string, [dom] real(64));
        fields["phi"] = phi;
        fields["VelocityX"] = uu;
        fields["graduX"] = graduX;
        fields["graduY"] = graduY;
        fields["VelocityY"] = vv;
        fields["gradvX"] = gradvX;
        fields["gradvY"] = gradvY;
        fields["VelocityZ"] = ww;
        fields["rho"] = rhorho;
        fields["gradRhoX"] = gradRhoX;
        fields["gradRhoY"] = gradRhoY;
        fields["cp"] = pp;
        fields["res"] = resres;
        fields["mach"] = machmach;
        fields["xElem"] = xElem;
        fields["yElem"] = yElem;
        fields["volumeElem"] = volumeElem;
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
            machWall[i] = this.mach(this.uFace_[face], this.vFace_[face], this.rhoFace_[face]);
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

        const wake_dom = {0..<this.wake_face_dom.size};
        var fieldsWake = new map(string, [wake_dom] real(64));
        var uWake: [wake_dom] real(64);
        var vWake: [wake_dom] real(64);
        var rhoWake: [wake_dom] real(64);
        var pWake: [wake_dom] real(64);
        var machWake: [wake_dom] real(64);
        var xWake: [wake_dom] real(64);
        var yWake: [wake_dom] real(64);
        var nxWake: [wake_dom] real(64);
        var nyWake: [wake_dom] real(64);
        var gammaWake: [wake_dom] real(64);

        forall (i, face) in zip(this.wake_face_dom, this.wakeFace_) {
            uWake[i-1] = this.uFace_[face];
            vWake[i-1] = this.vFace_[face];
            rhoWake[i-1] = this.rhoFace_[face];
            pWake[i-1] = (this.rhoFace_[face]**this.inputs_.GAMMA_ / (this.inputs_.GAMMA_ * this.inputs_.MACH_ * this.inputs_.MACH_ * this.inputs_.P_INF_) - 1 ) / (this.inputs_.GAMMA_/2*this.inputs_.MACH_**2);
            machWake[i-1] = this.machFace_[face];
            xWake[i-1] = this.faceCentroidX_[face];
            yWake[i-1] = this.faceCentroidY_[face];
            nxWake[i-1] = this.faceNormalX_[face];
            nyWake[i-1] = this.faceNormalY_[face];
            gammaWake[i-1] = this.circulation_;
        }

        fieldsWake["uWake"] = uWake;
        fieldsWake["vWake"] = vWake;
        fieldsWake["rhoWake"] = rhoWake;
        fieldsWake["cpWake"] = pWake;
        fieldsWake["machWake"] = machWake;
        fieldsWake["xWake"] = xWake;
        fieldsWake["yWake"] = yWake;
        fieldsWake["nxWake"] = nxWake;
        fieldsWake["nyWake"] = nyWake;
        fieldsWake["gammaWake"] = gammaWake;

        writer.writeWakeToCGNS(this.wakeFaceX_, this.wakeFaceY_, this.wakeFaceZ_, wake_dom, fieldsWake);
    }
}

}
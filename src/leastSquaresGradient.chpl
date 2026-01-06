/*
 * Least-Squares Gradient Reconstruction Module
 * 
 * This module provides a class that precomputes geometric coefficients for 
 * least-squares gradient reconstruction on unstructured meshes. The coefficients
 * depend only on mesh geometry (cell centroids, connectivity) and can be reused
 * for any scalar field.
 *
 * Mathematical formulation:
 * For each cell i with neighbors j, we minimize:
 *   sum_j w_j * (phi_j - phi_i - grad·(x_j - x_i))^2
 *
 * This leads to the normal equations:
 *   | sum(w*dx*dx)  sum(w*dx*dy) | | gradX |   | sum(w*dx*dphi) |
 *   | sum(w*dx*dy)  sum(w*dy*dy) | | gradY | = | sum(w*dy*dphi) |
 *
 * The left-hand-side matrix depends only on geometry and can be precomputed/inverted.
 * At runtime, we only need to compute the RHS and apply the stored inverse.
 */

module leastSquaresGradient {

use mesh;
use Math;
use Map;

/*
 * LeastSquaresGradient class
 *
 * Stores precomputed inverse matrix coefficients for each cell and 
 * weighted displacement vectors for each cell-neighbor pair.
 */
class LeastSquaresGradient {
    var mesh_: shared MeshData;
    var nelemDomain_: int;
    
    // Per-cell inverse matrix coefficients (symmetric 2x2 inverse)
    // For grad = invA * b, where invA = (1/det) * | a22  -a12 |
    //                                             | -a12  a11 |
    var elem_dom: domain(1) = {1..0};
    var invA11_: [elem_dom] real(64);  // a22 / det
    var invA12_: [elem_dom] real(64);  // -a12 / det
    var invA22_: [elem_dom] real(64);  // a11 / det
    
    // Per-face weighted displacement coefficients (w * dx, w * dy)
    // Stored per face for efficient RHS assembly
    var face_dom: domain(1) = {1..0};
    var wdx_: [face_dom] real(64);     // w * dx (from elem1 to elem2)
    var wdy_: [face_dom] real(64);     // w * dy (from elem1 to elem2)
    
    /*
     * Initialize the least-squares gradient operator
     */
    proc init(Mesh: shared MeshData, 
              ref elemCentroidX: [] real(64), 
              ref elemCentroidY: [] real(64)) {
        this.mesh_ = Mesh;
        this.nelemDomain_ = Mesh.nelem_;
        this.elem_dom = {1..this.nelemDomain_};
        this.face_dom = {1..Mesh.nedge_};
    }
    
    /*
     * Precompute all geometric coefficients
     * Call this once after mesh metrics are initialized
     */
    proc precompute(ref elemCentroidX: [] real(64), 
                    ref elemCentroidY: [] real(64)) {
        
        // Phase 1: Compute and store weighted displacements per face
        forall face in this.face_dom {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            // Displacement from elem1 centroid to elem2 centroid
            const dx = elemCentroidX[elem2] - elemCentroidX[elem1];
            const dy = elemCentroidY[elem2] - elemCentroidY[elem1];
            
            // Inverse-distance-squared weight
            const d2 = dx*dx + dy*dy;
            const w = 1.0 / d2;
            
            this.wdx_[face] = w * dx;
            this.wdy_[face] = w * dy;
        }
        
        // Phase 2: Assemble and invert the 2x2 matrix for each cell
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. 
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];
            
            // Accumulate matrix coefficients
            var a11 = 0.0, a12 = 0.0, a22 = 0.0;
            
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                
                // Get stored weighted displacement (always stored from elem1 to elem2)
                var wdx = this.wdx_[face];
                var wdy = this.wdy_[face];
                
                // If we are elem2, displacement is reversed
                if elem1 != elem {
                    wdx = -wdx;
                    wdy = -wdy;
                }
                
                // The weight w is |wdx|/|dx| = 1/d^2, and wdx = w*dx
                // So contributions to matrix are: wdx*dx = w*dx*dx, etc.
                // We can recover w*dx*dx = wdx * (wdx * d^2) = wdx^2 * d^2
                // But simpler: a11 += wdx * dx where dx = wdx / w... 
                // Actually easier to just recompute dx from wdx:
                // Since wdx = w*dx and w = 1/d^2 = 1/(dx^2+dy^2), we have
                // wdx * wdx + wdy * wdy = w^2 * d^2 = 1/d^2
                // So a11 = sum(w * dx * dx) = sum(wdx * dx)
                // And dx = wdx / w, but w = (wdx^2 + wdy^2) / (wdx*dx + wdy*dy) ... complicated
                // 
                // Simpler approach: store w separately or recompute from centroids
                // For efficiency, let's recompute dx, dy, w here since we're only doing this once
                
                const cx = elemCentroidX[elem];
                const cy = elemCentroidY[elem];
                const neighbor = if elem1 == elem then this.mesh_.edge2elem_[2, face] 
                                                  else this.mesh_.edge2elem_[1, face];
                
                const dx = elemCentroidX[neighbor] - cx;
                const dy = elemCentroidY[neighbor] - cy;
                const d2 = dx*dx + dy*dy;
                const w = 1.0 / d2;
                
                a11 += w * dx * dx;
                a12 += w * dx * dy;
                a22 += w * dy * dy;
            }
            
            // Compute and store inverse matrix coefficients
            const det = a11 * a22 - a12 * a12;
            
            if abs(det) > 1e-30 {
                const invDet = 1.0 / det;
                this.invA11_[elem] = a22 * invDet;   // (A^-1)_11 = a22/det
                this.invA12_[elem] = -a12 * invDet;  // (A^-1)_12 = -a12/det
                this.invA22_[elem] = a11 * invDet;   // (A^-1)_22 = a11/det
            } else {
                // Degenerate case (shouldn't happen for valid meshes)
                this.invA11_[elem] = 0.0;
                this.invA12_[elem] = 0.0;
                this.invA22_[elem] = 0.0;
            }
        }
    }
    
    /*
     * Compute gradient of a scalar field using precomputed coefficients
     * 
     * This is the fast runtime method - only computes the RHS and applies
     * the precomputed inverse matrix.
     */
    proc computeGradient(ref phi: [] real(64), 
                         ref gradX: [] real(64), 
                         ref gradY: [] real(64),
                         ref elemCentroidX: [] real(64),
                         ref elemCentroidY: [] real(64)) {
        
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. 
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];
            
            const phiI = phi[elem];
            const cx = elemCentroidX[elem];
            const cy = elemCentroidY[elem];
            
            // Accumulate RHS: b = sum(w * delta * dphi)
            var b1 = 0.0, b2 = 0.0;
            
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;
                
                const dphi = phi[neighbor] - phiI;
                
                // Recompute weighted displacement (could also store per-cell-face)
                const dx = elemCentroidX[neighbor] - cx;
                const dy = elemCentroidY[neighbor] - cy;
                const d2 = dx*dx + dy*dy;
                const w = 1.0 / d2;
                
                b1 += w * dx * dphi;
                b2 += w * dy * dphi;
            }
            
            // Apply precomputed inverse: grad = invA * b
            gradX[elem] = this.invA11_[elem] * b1 + this.invA12_[elem] * b2;
            gradY[elem] = this.invA12_[elem] * b1 + this.invA22_[elem] * b2;
        }
    }
}

/*
 * QR-Based Least-Squares Gradient Reconstruction (Blazek formulation)
 * 
 * This class implements the least-squares gradient using QR factorization
 * via Gram-Schmidt orthogonalization, as described in:
 *   Blazek, "Computational Fluid Dynamics: Principles and Applications"
 *   Section 5.3 (Eqs. 5.54-5.63)
 *
 * The method solves the over-determined system A*grad = b using:
 *   grad = R^(-1) * Q^T * b
 *
 * For 2D, the matrix A (NA x 2) with rows [theta_j*dx_ij, theta_j*dy_ij]
 * is decomposed as A = Q * R where:
 *   R = | r11  r12 |    (upper triangular)
 *       |  0   r22 |
 *
 * The gradient is computed as a weighted sum (Eq. 5.60):
 *   grad_i = sum_j w_ij * theta_j * (U_j - U_i)
 *
 * where w_ij are precomputed weight vectors that depend only on geometry.
 *
 * OPTIMIZATION: The final weights (wx*theta, wy*theta) are precomputed
 * and stored per face, for BOTH elem1's and elem2's perspectives.
 * At runtime, the gradient is simply:
 *   grad_i = sum_j [wxFinal_ij, wyFinal_ij] * (phi_j - phi_i)
 *
 * Advantages over normal equations approach:
 *   - Better numerical conditioning on stretched grids
 *   - Avoids forming A^T*A which can square the condition number
 *
 * The weighting theta_j = 1/d_ij (inverse distance) is recommended by Blazek
 * for accurate gradients on highly stretched/curved grids.
 */
class LeastSquaresGradientQR {
    var mesh_: shared MeshData;
    var nelemDomain_: int;

    var elem_dom: domain(1) = {1..0};
    var sumWx_: [elem_dom] real(64);  // Sum of wx weights per elem (for Jacobian)
    var sumWy_: [elem_dom] real(64);  // Sum of wy weights per elem
    
    // Precomputed final weights per face, for both perspectives
    // wxFinal1_[face] = weight for elem1 computing gradient using elem2 as neighbor
    // wxFinal2_[face] = weight for elem2 computing gradient using elem1 as neighbor
    var face_dom: domain(1) = {1..0};
    var wxFinal1_: [face_dom] real(64);  // For elem1's gradient
    var wyFinal1_: [face_dom] real(64);
    var wxFinal2_: [face_dom] real(64);  // For elem2's gradient
    var wyFinal2_: [face_dom] real(64);
    
    /*
     * Initialize the QR-based least-squares gradient operator
     */
    proc init(Mesh: shared MeshData) {
        this.mesh_ = Mesh;
        this.nelemDomain_ = Mesh.nelem_;
        this.elem_dom = {1..this.nelemDomain_};
        this.face_dom = {1..Mesh.nedge_};
    }
    
    /*
     * Precompute QR factorization coefficients and final weights
     * Call this once after mesh metrics are initialized
     *
     * Following Blazek Eqs. (5.55), (5.59), (5.61)-(5.62) adapted for 2D.
     * 
     * This precomputes everything so that at runtime, gradient = sum(w * dphi)
     */
    proc precompute(const ref elemCentroidX: [] real(64), 
                    const ref elemCentroidY: [] real(64)) {
        
        // Temporary storage for per-face theta and per-cell R matrix
        var theta: [this.face_dom] real(64);
        
        const elem_dom = {1..this.nelemDomain_};
        var r11: [elem_dom] real(64);
        var r12: [elem_dom] real(64);
        var r22: [elem_dom] real(64);
        
        // Phase 1: Compute theta (inverse distance) per face
        forall face in this.face_dom {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            
            const dx = elemCentroidX[elem2] - elemCentroidX[elem1];
            const dy = elemCentroidY[elem2] - elemCentroidY[elem1];
            const d = sqrt(dx*dx + dy*dy);
            
            theta[face] = 1.0 / d;
        }
        
        // Phase 2: Compute R matrix entries for each cell
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. 
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];
            
            const cx = elemCentroidX[elem];
            const cy = elemCentroidY[elem];
            
            // Accumulate sums for R matrix (Eq. 5.59)
            var sum_tdx_sq = 0.0;
            var sum_tdx_tdy = 0.0;
            var sum_tdy_sq = 0.0;
            
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;
                
                const dx = elemCentroidX[neighbor] - cx;
                const dy = elemCentroidY[neighbor] - cy;
                const t = theta[face];
                
                const tdx = t * dx;
                const tdy = t * dy;
                
                sum_tdx_sq  += tdx * tdx;
                sum_tdx_tdy += tdx * tdy;
                sum_tdy_sq  += tdy * tdy;
            }
            
            // R matrix entries (Eq. 5.59 for 2D)
            r11[elem] = sqrt(sum_tdx_sq);
            r12[elem] = sum_tdx_tdy / r11[elem];
            const r22_sq = sum_tdy_sq - r12[elem] * r12[elem];
            r22[elem] = sqrt(max(r22_sq, 1e-30));
        }
        
        // Phase 3: Compute and store final weights for each face
        // Store weights for both elem1's and elem2's perspectives
        forall face in this.face_dom {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];
            const t = theta[face];
            
            // Displacement from elem1 to elem2
            const dx12 = elemCentroidX[elem2] - elemCentroidX[elem1];
            const dy12 = elemCentroidY[elem2] - elemCentroidY[elem1];
            
            // Weights for elem1's gradient (neighbor is elem2)
            if elem1 <= this.nelemDomain_ {
                const r11_e = r11[elem1];
                const r12_e = r12[elem1];
                const r22_e = r22[elem1];
                
                const r12_r11 = r12_e / r11_e;
                const inv_r11_sq = 1.0 / (r11_e * r11_e);
                const inv_r22_sq = 1.0 / (r22_e * r22_e);
                
                // dx, dy from elem1's perspective (toward elem2)
                const dx = dx12;
                const dy = dy12;
                
                const alpha1 = t * dx * inv_r11_sq;
                const alpha2 = t * (dy - r12_r11 * dx) * inv_r22_sq;
                const wx = alpha1 - r12_r11 * alpha2;
                const wy = alpha2;
                
                this.wxFinal1_[face] = wx * t;
                this.wyFinal1_[face] = wy * t;
            }
            
            // Weights for elem2's gradient (neighbor is elem1)
            if elem2 <= this.nelemDomain_ {
                const r11_e = r11[elem2];
                const r12_e = r12[elem2];
                const r22_e = r22[elem2];
                
                const r12_r11 = r12_e / r11_e;
                const inv_r11_sq = 1.0 / (r11_e * r11_e);
                const inv_r22_sq = 1.0 / (r22_e * r22_e);
                
                // dx, dy from elem2's perspective (toward elem1) = -dx12, -dy12
                const dx = -dx12;
                const dy = -dy12;
                
                const alpha1 = t * dx * inv_r11_sq;
                const alpha2 = t * (dy - r12_r11 * dx) * inv_r22_sq;
                const wx = alpha1 - r12_r11 * alpha2;
                const wy = alpha2;
                
                this.wxFinal2_[face] = wx * t;
                this.wyFinal2_[face] = wy * t;
            }
        }

        // Compute sum of final weights for Jacobian computation
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. 
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];
            
            var gx = 0.0, gy = 0.0;         
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                if elem == elem1 {
                    gx += this.wxFinal1_[face];
                    gy += this.wyFinal1_[face];
                } else {
                    gx += this.wxFinal2_[face];
                    gy += this.wyFinal2_[face];
                }
            }

            this.sumWx_[elem] = gx;
            this.sumWy_[elem] = gy;
        }
    }
    
    /*
     * Compute gradient of a scalar field using fully precomputed weights
     */
    proc computeGradient(const ref phi: [] real(64), 
                         ref gradX: [] real(64), 
                         ref gradY: [] real(64),
                         const ref kuttaCell_: [] int,
                         const circulation_: real(64)) {
        
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. 
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];
            
            const phiI = phi[elem];
            var gx = 0.0, gy = 0.0;
            
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                var phiJ = phi[neighbor];

                const elemKuttaType = kuttaCell_[elem];
                const neighborKuttaType = kuttaCell_[neighbor];
                if (elemKuttaType == 1 && neighborKuttaType == -1) {
                    // Elem is above wake, neighbor is below wake
                    // To get continuous potential, add Γ to lower surface value
                    // φ_seen = φ_lower + Γ = φ_upper
                    phiJ += circulation_;
                } else if (elemKuttaType == -1 && neighborKuttaType == 1) {
                    // Elem is below wake, neighbor is above wake
                    // To get continuous potential, subtract Γ from upper surface value
                    // φ_seen = φ_upper - Γ = φ_lower
                    phiJ -= circulation_;
                }

                const dphi = phiJ - phiI;
                
                // Use precomputed weights for correct perspective
                if elem == elem1 {
                    gx += this.wxFinal1_[face] * dphi;
                    gy += this.wyFinal1_[face] * dphi;
                } else {
                    gx += this.wxFinal2_[face] * dphi;
                    gy += this.wyFinal2_[face] * dphi;
                }
            }
            
            gradX[elem] = gx;
            gradY[elem] = gy;
        }
    }

    /*
     * Compute gradient of a scalar field using fully precomputed weights
     */
    proc computeGradient(const ref phi: [] real(64), 
                         ref gradX: [] real(64), 
                         ref gradY: [] real(64),
                         const ref kuttaCell_: [] int,
                         const ref wakeFace2index: map(int, int),
                         const ref wakeFaceGamma: [] real(64)) {
        
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. 
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];
            
            const phiI = phi[elem];
            var gx = 0.0, gy = 0.0;
            
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                var phiJ = phi[neighbor];

                const elemKuttaType = kuttaCell_[elem];
                const neighborKuttaType = kuttaCell_[neighbor];
                if (elemKuttaType == 1 && neighborKuttaType == -1) {
                    // Elem is above wake, neighbor is below wake
                    // To get continuous potential, add Γ to lower surface value
                    // φ_seen = φ_lower + Γ = φ_upper
                    try {
                        const wakeIndex = wakeFace2index[face];
                        const gamma = wakeFaceGamma[wakeIndex];
                        phiJ -= gamma;
                    }
                    catch e: Error {
                        halt("Error: Face ", face, " not found in wakeFace2index map.");
                    }
                } else if (elemKuttaType == -1 && neighborKuttaType == 1) {
                    // Elem is below wake, neighbor is above wake
                    // To get continuous potential, subtract Γ from upper surface value
                    // φ_seen = φ_upper - Γ = φ_lower
                    try {
                        const wakeIndex = wakeFace2index[face];
                        const gamma = wakeFaceGamma[wakeIndex];
                        phiJ += gamma;
                    }
                    catch e: Error {
                        halt("Error: Face ", face, " not found in wakeFace2index map.");
                    }
                }

                const dphi = phiJ - phiI;
                
                // Use precomputed weights for correct perspective
                if elem == elem1 {
                    gx += this.wxFinal1_[face] * dphi;
                    gy += this.wyFinal1_[face] * dphi;
                } else {
                    gx += this.wxFinal2_[face] * dphi;
                    gy += this.wyFinal2_[face] * dphi;
                }
            }
            
            gradX[elem] = gx;
            gradY[elem] = gy;
        }
    }

    /*
     * Compute gradient of a scalar field using fully precomputed weights
     */
    proc computeGradient(const ref phi: [] real(64), 
                         ref gradX: [] real(64), 
                         ref gradY: [] real(64)) {
        
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 .. 
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];
            
            const phiI = phi[elem];
            var gx = 0.0, gy = 0.0;
            
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                const dphi = phi[neighbor] - phiI;
                
                // Use precomputed weights for correct perspective
                if elem == elem1 {
                    gx += this.wxFinal1_[face] * dphi;
                    gy += this.wyFinal1_[face] * dphi;
                } else {
                    gx += this.wxFinal2_[face] * dphi;
                    gy += this.wyFinal2_[face] * dphi;
                }
            }
            
            gradX[elem] = gx;
            gradY[elem] = gy;
        }
    }
}

/*
 * Pseudo-Laplacian Operator (Blazek formulation)
 * 
 * This class implements the pseudo-Laplacian operator for artificial dissipation
 * on unstructured meshes, as described in:
 *   Blazek, "Computational Fluid Dynamics: Principles and Applications"
 *   Section 5.3.1 (Eqs. 5.24-5.30)
 *
 * The pseudo-Laplacian is defined as (Eq. 5.24):
 *   L(U_I) = Σ_J θ_IJ * (U_J - U_I)
 *
 * where θ_IJ are distance-weighted geometric weights that ensure the
 * pseudo-Laplacian vanishes for a linearly varying function on any grid.
 *
 * The weights are computed as (Eqs. 5.25-5.26):
 *   θ_IJ = 1 + θ'_IJ
 *   θ'_IJ = λ_x * (x_J - x_I) + λ_y * (y_J - y_I)
 *
 * The Lagrange multipliers λ are computed for each cell from the first-order
 * and second-order moments of the cell's neighbors (Eqs. 5.27-5.30).
 *
 * For 2D, the system simplifies to:
 *   |I_xx  I_xy| |λ_x|   |R_x|
 *   |I_xy  I_yy| |λ_y| = |R_y|
 *
 * Applications:
 * - Second-order artificial dissipation: k^(2) * Σ θ_IJ * (W_J - W_I)
 * - Fourth-order artificial dissipation: L(L(W)) (Laplacian of Laplacian)
 * - JST (Jameson-Schmidt-Turkel) scheme stabilization
 */
class PseudoLaplacian {
    var mesh_: shared MeshData;
    var nelemDomain_: int;
    var nface_: int;

    // Per-cell Lagrange multipliers
    var elem_dom: domain(1) = {1..0};
    var lambdaX_: [elem_dom] real(64);
    var lambdaY_: [elem_dom] real(64);

    // Precomputed weights per face (for both elem1 and elem2 perspectives)
    // theta1_[face] = weight when computing L(U) for elem1
    // theta2_[face] = weight when computing L(U) for elem2
    var face_dom: domain(1) = {1..0};
    var theta1_: [face_dom] real(64);
    var theta2_: [face_dom] real(64);

    // Sum of weights per cell (for efficient computation)
    var sumTheta_: [elem_dom] real(64);

    /*
     * Initialize the pseudo-Laplacian operator
     */
    proc init(Mesh: shared MeshData) {
        this.mesh_ = Mesh;
        this.nelemDomain_ = Mesh.nelem_;
        this.nface_ = Mesh.nedge_;
        this.elem_dom = {1..this.nelemDomain_};
        this.face_dom = {1..this.nface_};
    }

    /*
     * Precompute Lagrange multipliers and weights
     * Call this once after mesh metrics are initialized
     *
     * Following Blazek Eqs. (5.27)-(5.30) for 2D
     */
    proc precompute(const ref elemCentroidX: [] real(64),
                    const ref elemCentroidY: [] real(64)) {
        
        // Phase 1: Compute Lagrange multipliers for each cell
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 ..
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];

            const cx = elemCentroidX[elem];
            const cy = elemCentroidY[elem];

            // First-order moments (Eq. 5.29)
            var Rx = 0.0, Ry = 0.0;

            // Second-order moments (Eq. 5.30)
            var Ixx = 0.0, Iyy = 0.0, Ixy = 0.0;

            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                const dx = elemCentroidX[neighbor] - cx;
                const dy = elemCentroidY[neighbor] - cy;

                // First-order moments
                Rx += dx;
                Ry += dy;

                // Second-order moments
                Ixx += dx * dx;
                Iyy += dy * dy;
                Ixy += dx * dy;
            }

            // Solve 2x2 system for Lagrange multipliers (Eq. 5.27 simplified for 2D)
            // For linear exactness, we need:
            //   Σ θ_IJ (x_J - x_I) = 0  and  Σ θ_IJ (y_J - y_I) = 0
            // Substituting θ_IJ = 1 + λ_x·Δx + λ_y·Δy and expanding:
            //   R_x + λ_x·I_xx + λ_y·I_xy = 0
            //   R_y + λ_x·I_xy + λ_y·I_yy = 0
            // Therefore:
            //   |Ixx  Ixy| |λx|   |-Rx|
            //   |Ixy  Iyy| |λy| = |-Ry|
            //
            // Using Cramer's rule:
            // det = Ixx*Iyy - Ixy^2
            // λx = (-Rx*Iyy + Ry*Ixy) / det
            // λy = (-Ry*Ixx + Rx*Ixy) / det

            const det = Ixx * Iyy - Ixy * Ixy;

            if abs(det) > 1e-30 {
                const invDet = 1.0 / det;
                this.lambdaX_[elem] = (-Rx * Iyy + Ry * Ixy) * invDet;
                this.lambdaY_[elem] = (-Ry * Ixx + Rx * Ixy) * invDet;
            } else {
                // Degenerate case (shouldn't happen for valid meshes)
                this.lambdaX_[elem] = 0.0;
                this.lambdaY_[elem] = 0.0;
            }
        }

        // Phase 2: Compute and store weights per face for both perspectives
        forall face in this.face_dom {
            const elem1 = this.mesh_.edge2elem_[1, face];
            const elem2 = this.mesh_.edge2elem_[2, face];

            // Displacement from elem1 to elem2
            const dx12 = elemCentroidX[elem2] - elemCentroidX[elem1];
            const dy12 = elemCentroidY[elem2] - elemCentroidY[elem1];

            // Weight for elem1's pseudo-Laplacian (neighbor is elem2)
            // θ_IJ = 1 + λ_x,I * dx + λ_y,I * dy (Eqs. 5.25-5.26)
            if elem1 <= this.nelemDomain_ {
                const thetaPrime = this.lambdaX_[elem1] * dx12 + this.lambdaY_[elem1] * dy12;
                // Clip to [0, 2] for stability on distorted grids (Blazek recommendation)
                this.theta1_[face] = max(0.0, min(2.0, 1.0 + thetaPrime));
            } else {
                this.theta1_[face] = 1.0;  // Ghost cell: use uniform weight
            }

            // Weight for elem2's pseudo-Laplacian (neighbor is elem1)
            // Note: displacement is -dx12, -dy12 from elem2's perspective
            if elem2 <= this.nelemDomain_ {
                const thetaPrime = this.lambdaX_[elem2] * (-dx12) + this.lambdaY_[elem2] * (-dy12);
                // Clip to [0, 2] for stability
                this.theta2_[face] = max(0.0, min(2.0, 1.0 + thetaPrime));
            } else {
                this.theta2_[face] = 1.0;  // Ghost cell: use uniform weight
            }
        }

        // Phase 3: Compute sum of weights per cell (for Laplacian of Laplacian)
        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 ..
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];
            var sumT = 0.0;
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                if elem == elem1 {
                    sumT += this.theta1_[face];
                } else {
                    sumT += this.theta2_[face];
                }
            }
            this.sumTheta_[elem] = sumT;
        }
    }

    /*
     * Compute the pseudo-Laplacian of a scalar field: L(U)
     * 
     * L(U_I) = Σ_J θ_IJ * (U_J - U_I)
     */
    proc apply(const ref U: [] real(64),
               ref LU: [] real(64)) {

        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 ..
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];

            const UI = U[elem];
            var sum = 0.0;

            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                // Get the correct weight for this perspective
                const theta = if elem == elem1 then this.theta1_[face] else this.theta2_[face];

                sum += theta * (U[neighbor] - UI);
            }

            LU[elem] = sum;
        }
    }

    /*
     * Compute the Laplacian of Laplacian: L(L(U)) for fourth-order dissipation
     *
     * This is done in two passes:
     * 1. Compute L(U) for all cells
     * 2. Compute L(L(U)) using the result
     *
     * The temp array is provided externally to avoid allocation overhead
     */
    proc applyLaplacianOfLaplacian(const ref U: [] real(64),
                                    ref LLU: [] real(64),
                                    ref temp: [] real(64)) {
        // First pass: L(U)
        this.apply(U, temp);

        // Second pass: L(L(U))
        this.apply(temp, LLU);
    }

    /*
     * Compute second-order artificial dissipation term (scaled by spectral radius)
     *
     * D^(2)_I = Σ_J (λ_c)_IJ * ε^(2)_IJ * θ_IJ * (W_J - W_I)
     *
     * where (λ_c)_IJ is the spectral radius and ε^(2)_IJ is the shock sensor
     */
    proc computeSecondOrderDissipation(const ref U: [] real(64),
                                        const ref spectralRadius: [] real(64),
                                        const ref epsilon2: [] real(64),
                                        ref D2: [] real(64)) {

        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 ..
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];

            const UI = U[elem];
            var sum = 0.0;

            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                // Spectral radius at face: average of cell values
                const lambdaC = 0.5 * (spectralRadius[elem] + spectralRadius[neighbor]);

                // Shock sensor: max of adjacent cells
                const eps2 = max(epsilon2[elem], epsilon2[neighbor]);

                // Weight for this perspective
                const theta = if elem == elem1 then this.theta1_[face] else this.theta2_[face];

                sum += lambdaC * eps2 * theta * (U[neighbor] - UI);
            }

            D2[elem] = sum;
        }
    }

    /*
     * Compute fourth-order artificial dissipation term
     *
     * D^(4)_I = -Σ_J (λ_c)_IJ * ε^(4)_IJ * θ_IJ * (L(W_J) - L(W_I))
     *
     * Requires pre-computed L(U) passed as LU
     */
    proc computeFourthOrderDissipation(const ref U: [] real(64),
                                        const ref LU: [] real(64),
                                        const ref spectralRadius: [] real(64),
                                        const ref epsilon4: [] real(64),
                                        ref D4: [] real(64)) {

        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 ..
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];

            var sum = 0.0;

            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                // Spectral radius at face
                const lambdaC = 0.5 * (spectralRadius[elem] + spectralRadius[neighbor]);

                // Fourth-order coefficient
                const eps4 = max(0.0, epsilon4[elem] - epsilon4[neighbor]);

                // Weight for this perspective
                const theta = if elem == elem1 then this.theta1_[face] else this.theta2_[face];

                sum += lambdaC * eps4 * theta * (LU[neighbor] - LU[elem]);
            }

            D4[elem] = -sum;  // Note the negative sign
        }
    }

    /*
     * Compute combined JST artificial dissipation (second + fourth order)
     *
     * D_I = Σ_J (λ_c)_IJ * θ_IJ * [ε^(2)_IJ * (W_J - W_I) - ε^(4)_IJ * (L(W_J) - L(W_I))]
     *
     * Parameters:
     *   k2: second-order coefficient (typically 0.5)
     *   k4: fourth-order coefficient (typically 1/64 to 1/128)
     */
    proc computeJSTDissipation(const ref U: [] real(64),
                                const ref spectralRadius: [] real(64),
                                const ref pressureSensor: [] real(64),
                                const k2: real(64),
                                const k4: real(64),
                                ref D: [] real(64),
                                ref tempLU: [] real(64)) {

        // First compute L(U) for fourth-order term
        this.apply(U, tempLU);

        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 ..
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];

            const UI = U[elem];
            const LUI = tempLU[elem];
            var sum = 0.0;

            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                // Spectral radius at face
                const lambdaC = 0.5 * (spectralRadius[elem] + spectralRadius[neighbor]);

                // Pressure sensor (Eq. 5.36): Υ = max of adjacent cells
                const upsilon = max(pressureSensor[elem], pressureSensor[neighbor]);

                // Coefficients (Eq. 5.35)
                const eps2 = k2 * upsilon;
                const eps4 = max(0.0, k4 - eps2);

                // Weight for this perspective
                const theta = if elem == elem1 then this.theta1_[face] else this.theta2_[face];

                // Combined dissipation
                sum += lambdaC * theta * (eps2 * (U[neighbor] - UI) 
                                         - eps4 * (tempLU[neighbor] - LUI));
            }

            D[elem] = sum;
        }
    }

    /*
     * Compute pressure sensor for JST scheme (Eq. 5.36)
     *
     * Υ_I = |Σ_J θ_IJ * (p_J - p_I)| / Σ_J (p_J + p_I)
     *
     * The sensor detects shocks based on second derivative of pressure
     */
    proc computePressureSensor(const ref pressure: [] real(64),
                                ref sensor: [] real(64)) {

        forall elem in 1..this.nelemDomain_ {
            const faces = this.mesh_.elem2edge_[this.mesh_.elem2edgeIndex_[elem] + 1 ..
                                                 this.mesh_.elem2edgeIndex_[elem + 1]];

            const pI = pressure[elem];
            var numerator = 0.0;
            var denominator = 0.0;

            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                const neighbor = if elem1 == elem then elem2 else elem1;

                const pJ = pressure[neighbor];

                // Weight for this perspective
                const theta = if elem == elem1 then this.theta1_[face] else this.theta2_[face];

                numerator += theta * (pJ - pI);
                denominator += pJ + pI;
            }

            if abs(denominator) > 1e-30 {
                sensor[elem] = abs(numerator) / denominator;
            } else {
                sensor[elem] = 0.0;
            }
        }
    }
}

}

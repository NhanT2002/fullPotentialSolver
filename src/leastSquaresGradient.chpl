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
    var sumWx1_: [elem_dom] real(64);  // Sum of wx weights per elem (for Jacobian)
    var sumWy1_: [elem_dom] real(64);  // Sum of wy weights per elem
    var sumWx2_: [elem_dom] real(64);  // Sum of wx weights per elem (for Jacobian)
    var sumWy2_: [elem_dom] real(64);  // Sum of wy weights per elem
    
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
    proc init(Mesh: shared MeshData, 
              ref elemCentroidX: [] real(64), 
              ref elemCentroidY: [] real(64)) {
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
    proc precompute(ref elemCentroidX: [] real(64), 
                    ref elemCentroidY: [] real(64)) {
        
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
            for face in faces {
                const elem1 = this.mesh_.edge2elem_[1, face];
                const elem2 = this.mesh_.edge2elem_[2, face];
                if elem == elem1 {
                    this.sumWx1_[elem] += this.wxFinal1_[face];
                    this.sumWy1_[elem] += this.wyFinal1_[face];
                } else {
                    this.sumWx2_[elem] += this.wxFinal2_[face];
                    this.sumWy2_[elem] += this.wyFinal2_[face];
                }
            }
        }
    }
    
    /*
     * Compute gradient of a scalar field using fully precomputed weights
     */
    proc computeGradient(ref phi: [] real(64), 
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

}

module testMetrics {
    use Math;
    use IO;
    use spatialDiscretization;
    use mesh;
    import input.potentialInputs;
    use linearAlgebra;

    // ============== METRIC VERIFICATION TESTS ==============

    proc run_tests() {
        var inputs = new potentialInputs();
        var Mesh = new shared MeshData(inputs.GRID_FILENAME_, inputs.ELEMENT_TYPE_);
        Mesh.buildConnectivity();
        var spatialDisc = new shared spatialDiscretization(Mesh, inputs);
        spatialDisc.initializeMetrics();
        spatialDisc.initializeSolution();
        verifyMetrics(spatialDisc);

        // 1) Residual from the initialized (freestream) velocity field.
        // This should be ~0 (roundoff) because div(U_inf) = 0 and each control volume is closed.
        spatialDisc.computeFluxes();
        spatialDisc.computeResiduals();
        {
            const res = RMSE(spatialDisc.res_, spatialDisc.elemVolume_);
            const resMax = max reduce abs(spatialDisc.res_);
            writeln("Freestream residual (before phi gradient): L2=", res, " max=", resMax);
        }

        // spatialDisc.updateGhostCells();
        
        // Test Green-Gauss gradient
        spatialDisc.computeVelocityFromPhi();
        spatialDisc.computeFluxes();
        spatialDisc.computeResiduals();
        {
            const elemDom = {1..spatialDisc.nelemDomain_};
            const uDevMax = max reduce abs(spatialDisc.uu_[elemDom] - inputs.U_INF_);
            const vDevMax = max reduce abs(spatialDisc.vv_[elemDom] - inputs.V_INF_);
            const res = RMSE(spatialDisc.res_, spatialDisc.elemVolume_);
            const resMax = max reduce abs(spatialDisc.res_);
            writeln("[Green-Gauss] velocity deviation: max|u-Uinf|=", uDevMax, " max|v-Vinf|=", vDevMax);
            writeln("[Green-Gauss] residual: RMSE=", res, " max=", resMax);
        }

        // Reinitialize velocity to freestream before testing Least-Squares
        spatialDisc.initializeSolution();
        
        // Test Least-Squares gradient
        spatialDisc.computeVelocityFromPhiLeastSquares();
        spatialDisc.computeFluxes();
        spatialDisc.computeResiduals();
        {
            const elemDom = {1..spatialDisc.nelemDomain_};
            const uDevMax = max reduce abs(spatialDisc.uu_[elemDom] - inputs.U_INF_);
            const vDevMax = max reduce abs(spatialDisc.vv_[elemDom] - inputs.V_INF_);
            const res = RMSE(spatialDisc.res_, spatialDisc.elemVolume_);
            const resMax = max reduce abs(spatialDisc.res_);
            writeln("[Least-Squares] velocity deviation: max|u-Uinf|=", uDevMax, " max|v-Vinf|=", vDevMax);
            writeln("[Least-Squares] residual: RMSE=", res, " max=", resMax);
        }

        // Reinitialize velocity to freestream before testing QR Least-Squares
        spatialDisc.initializeSolution();
        
        // Test QR-based Least-Squares gradient (Blazek formulation)
        spatialDisc.computeVelocityFromPhiLeastSquaresQR();
        spatialDisc.computeFluxes();
        spatialDisc.computeResiduals();
        {
            const elemDom = {1..spatialDisc.nelemDomain_};
            const uDevMax = max reduce abs(spatialDisc.uu_[elemDom] - inputs.U_INF_);
            const vDevMax = max reduce abs(spatialDisc.vv_[elemDom] - inputs.V_INF_);
            const res = RMSE(spatialDisc.res_, spatialDisc.elemVolume_);
            const resMax = max reduce abs(spatialDisc.res_);
            writeln("[Least-Squares QR] velocity deviation: max|u-Uinf|=", uDevMax, " max|v-Vinf|=", vDevMax);
            writeln("[Least-Squares QR] residual: RMSE=", res, " max=", resMax);
        }

        spatialDisc.writeSolution();
    }
    
    proc verifyMetrics(ref disc: spatialDiscretization): bool {
        var allPassed = true;
        
        writeln("=== Verifying Computed Metrics ===");
        
        if !testElementCentroids(disc) then allPassed = false;
        if !testElementVolumes(disc) then allPassed = false;
        if !testFaceCentroids(disc) then allPassed = false;
        if !testFaceAreas(disc) then allPassed = false;
        if !testFaceNormals(disc) then allPassed = false;
        if !testGhostCellCentroids(disc) then allPassed = false;
        
        if allPassed {
            writeln("=== All metric tests PASSED ===");
        } else {
            writeln("=== Some metric tests FAILED ===");
        }
        
        return allPassed;
    }
    
    proc testElementCentroids(ref disc: spatialDiscretization): bool {
        const tol = 1e-12;
        var maxErr = 0.0;
        var failCount = 0;
        
        for elem in 1..disc.nelemDomain_ {
            const nodeStart = disc.mesh_.elem2nodeIndex_[elem] + 1;
            const nodeEnd = disc.mesh_.elem2nodeIndex_[elem + 1];
            const nodes = disc.mesh_.elem2node_[nodeStart..nodeEnd];
            
            var expectedX = 0.0, expectedY = 0.0;
            for node in nodes {
                expectedX += disc.mesh_.X_[node];
                expectedY += disc.mesh_.Y_[node];
            }
            expectedX /= nodes.size;
            expectedY /= nodes.size;
            
            const errX = abs(disc.elemCentroidX_[elem] - expectedX);
            const errY = abs(disc.elemCentroidY_[elem] - expectedY);
            const err = max(errX, errY);
            
            if err > maxErr then maxErr = err;
            if err > tol {
                failCount += 1;
                if failCount <= 3 {
                    writeln("  [FAIL] Element ", elem, " centroid: expected (", 
                            expectedX, ", ", expectedY, "), got (",
                            disc.elemCentroidX_[elem], ", ", disc.elemCentroidY_[elem], ")");
                }
            }
        }
        
        const passed = (failCount == 0);
        writeln("[", if passed then "PASS" else "FAIL", "] Element centroids - max error: ", maxErr, 
                if failCount > 0 then " (" + failCount:string + " failures)" else "");
        return passed;
    }
    
    proc testElementVolumes(ref disc: spatialDiscretization): bool {
        var failCount = 0;
        var minVol = max(real(64));
        var totalVol = 0.0;
        
        for elem in 1..disc.nelemDomain_ {
            const vol = disc.elemVolume_[elem];
            
            if vol <= 0.0 {
                failCount += 1;
                if failCount <= 3 {
                    writeln("  [FAIL] Element ", elem, " has non-positive volume: ", vol);
                }
            }
            
            if vol < minVol then minVol = vol;
            totalVol += vol;
        }
        
        const passed = (failCount == 0);
        writeln("[", if passed then "PASS" else "FAIL", "] Element volumes - min: ", minVol, 
                ", total: ", totalVol,
                if failCount > 0 then " (" + failCount:string + " non-positive)" else "");
        return passed;
    }
    
    proc testFaceCentroids(ref disc: spatialDiscretization): bool {
        const tol = 1e-12;
        var maxErr = 0.0;
        var failCount = 0;
        
        for face in 1..disc.nface_ {
            const node1 = disc.mesh_.edge2node_[1, face];
            const node2 = disc.mesh_.edge2node_[2, face];
            
            const expectedX = (disc.mesh_.X_[node1] + disc.mesh_.X_[node2]) * 0.5;
            const expectedY = (disc.mesh_.Y_[node1] + disc.mesh_.Y_[node2]) * 0.5;
            
            const errX = abs(disc.faceCentroidX_[face] - expectedX);
            const errY = abs(disc.faceCentroidY_[face] - expectedY);
            const err = max(errX, errY);
            
            if err > maxErr then maxErr = err;
            if err > tol then failCount += 1;
        }
        
        const passed = (failCount == 0);
        writeln("[", if passed then "PASS" else "FAIL", "] Face centroids - max error: ", maxErr,
                if failCount > 0 then " (" + failCount:string + " failures)" else "");
        return passed;
    }
    
    proc testFaceAreas(ref disc: spatialDiscretization): bool {
        const tol = 1e-12;
        var maxErr = 0.0;
        var failCount = 0;
        var minArea = max(real(64));
        
        for face in 1..disc.nface_ {
            const node1 = disc.mesh_.edge2node_[1, face];
            const node2 = disc.mesh_.edge2node_[2, face];
            
            const dx = disc.mesh_.X_[node2] - disc.mesh_.X_[node1];
            const dy = disc.mesh_.Y_[node2] - disc.mesh_.Y_[node1];
            const expectedArea = sqrt(dx*dx + dy*dy);
            
            const err = abs(disc.faceArea_[face] - expectedArea);
            if err > maxErr then maxErr = err;
            if err > tol then failCount += 1;
            
            if disc.faceArea_[face] < minArea then minArea = disc.faceArea_[face];
            
            if disc.faceArea_[face] <= 0.0 {
                writeln("  [FAIL] Face ", face, " has non-positive area: ", disc.faceArea_[face]);
            }
        }
        
        const passed = (failCount == 0 && minArea > 0.0);
        writeln("[", if passed then "PASS" else "FAIL", "] Face areas - max error: ", maxErr, 
                ", min area: ", minArea,
                if failCount > 0 then " (" + failCount:string + " failures)" else "");
        return passed;
    }
    
    proc testFaceNormals(ref disc: spatialDiscretization): bool {
        const tol = 1e-10;
        var maxMagErr = 0.0;
        var maxDotErr = 0.0;
        var magFailCount = 0;
        var perpFailCount = 0;
        var dirFailCount = 0;
        
        for face in 1..disc.nface_ {
            const nx = disc.faceNormalX_[face];
            const ny = disc.faceNormalY_[face];
            
            // Test 1: Unit magnitude
            const mag = sqrt(nx*nx + ny*ny);
            const magErr = abs(mag - 1.0);
            if magErr > maxMagErr then maxMagErr = magErr;
            if magErr > tol then magFailCount += 1;
            
            // Test 2: Perpendicular to edge
            const node1 = disc.mesh_.edge2node_[1, face];
            const node2 = disc.mesh_.edge2node_[2, face];
            const dx = disc.mesh_.X_[node2] - disc.mesh_.X_[node1];
            const dy = disc.mesh_.Y_[node2] - disc.mesh_.Y_[node1];
            const d = sqrt(dx*dx + dy*dy);
            const tx = dx / d;  // tangent
            const ty = dy / d;
            const dotErr = abs(nx*tx + ny*ty);  // should be 0
            if dotErr > maxDotErr then maxDotErr = dotErr;
            if dotErr > tol then perpFailCount += 1;
            
            // Test 3: Normal points away from elem1 centroid
            const elem1 = disc.mesh_.edge2elem_[1, face];
            const fcx = disc.faceCentroidX_[face];
            const fcy = disc.faceCentroidY_[face];
            const toCentroidX = disc.elemCentroidX_[elem1] - fcx;
            const toCentroidY = disc.elemCentroidY_[elem1] - fcy;
            const dotProduct = nx * toCentroidX + ny * toCentroidY;
            
            // Normal should point away from elem1, so dot product should be <= 0
            if dotProduct > tol {
                dirFailCount += 1;
                if dirFailCount <= 3 {
                    writeln("  [FAIL] Face ", face, " normal points toward elem1 (dot=", dotProduct, ")");
                }
            }
        }
        
        const passed = (magFailCount == 0 && perpFailCount == 0 && dirFailCount == 0);
        writeln("[", if passed then "PASS" else "FAIL", "] Face normals:");
        writeln("    Unit magnitude - max error: ", maxMagErr, 
                if magFailCount > 0 then " (" + magFailCount:string + " failures)" else "");
        writeln("    Perpendicularity - max error: ", maxDotErr,
                if perpFailCount > 0 then " (" + perpFailCount:string + " failures)" else "");
        writeln("    Direction - ", if dirFailCount == 0 then "OK" else dirFailCount:string + " point wrong way");
        return passed;
    }
    
    proc testGhostCellCentroids(ref disc: spatialDiscretization): bool {
        const tol = 1e-10;
        var failCount = 0;
        var maxDistErr = 0.0;
        
        // Helper to test mirroring for a boundary face
        proc testMirror(face: int): bool {
            const elem1 = disc.mesh_.edge2elem_[1, face];
            const elem2 = disc.mesh_.edge2elem_[2, face];
            
            const (interiorElem, ghostElem) = 
                if elem1 <= disc.nelemDomain_ then (elem1, elem2) else (elem2, elem1);
            
            // Get face line
            const node1 = disc.mesh_.edge2node_[1, face];
            const node2 = disc.mesh_.edge2node_[2, face];
            const x1 = disc.mesh_.X_[node1];
            const y1 = disc.mesh_.Y_[node1];
            const x2 = disc.mesh_.X_[node2];
            const y2 = disc.mesh_.Y_[node2];
            
            const dx = x2 - x1;
            const dy = y2 - y1;
            const M = sqrt(dx*dx + dy*dy);
            const A = dy / M;
            const B = -dx / M;
            const C = -A*x1 - B*y1;
            
            // Distance from interior centroid to line
            const intX = disc.elemCentroidX_[interiorElem];
            const intY = disc.elemCentroidY_[interiorElem];
            const distInt = A*intX + B*intY + C;
            
            // Distance from ghost centroid to line
            const ghostX = disc.elemCentroidX_[ghostElem];
            const ghostY = disc.elemCentroidY_[ghostElem];
            const distGhost = A*ghostX + B*ghostY + C;
            
            // Ghost should be at same distance but opposite side
            const distErr = abs(distInt + distGhost);
            if distErr > maxDistErr then maxDistErr = distErr;
            
            // Midpoint of interior and ghost should lie on the face line
            const midX = (intX + ghostX) * 0.5;
            const midY = (intY + ghostY) * 0.5;
            const midDist = abs(A*midX + B*midY + C);
            
            return (distErr < tol && midDist < tol);
        }
        
        for face in disc.mesh_.edgeWall_ {
            if !testMirror(face) then failCount += 1;
        }
        for face in disc.mesh_.edgeFarfield_ {
            if !testMirror(face) then failCount += 1;
        }
        
        const totalBoundaryFaces = disc.mesh_.edgeWall_.size + disc.mesh_.edgeFarfield_.size;
        const passed = (failCount == 0);
        writeln("[", if passed then "PASS" else "FAIL", "] Ghost cell centroids - tested ", 
                totalBoundaryFaces, " boundary faces, max distance error: ", maxDistErr,
                if failCount > 0 then " (" + failCount:string + " failures)" else "");
        return passed;
    }
}

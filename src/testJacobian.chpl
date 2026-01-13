module testJacobian
{
use mesh;
use Math;
use linearAlgebra;
use Time;
import input.potentialInputs;
use unsteadySpatialDiscretization;
use unsteadyTemporalDiscretization;
use PETSCapi;
use C_PETSC;
use petsc;
use CTypes;

// ============================================================================
// Jacobian Verification using Finite Differences
// Compares analytical Jacobian entries with FD approximations
// ============================================================================

proc verifyJacobian(ref temporalDisc: unsteadyTemporalDiscretization) {
    writeln("\n=== JACOBIAN VERIFICATION ===\n");
    
    const spatialDisc = temporalDisc.spatialDisc_;
    const N = spatialDisc.nelemDomain_;
    const nWake = spatialDisc.wake_face_dom.size;
    const totalDOF = 2*N + nWake;
    
    writeln("DOF breakdown: N=", N, ", nWake=", nWake, ", total=", totalDOF);
    
    // Compute the analytical Jacobian
    temporalDisc.computeGradientSensitivity();
    temporalDisc.computeJacobian();
    
    // Step size for finite differences
    const eps = 1.0e-7;
    
    // Store base state
    var rho_base: [1..N] real(64);
    var phi_base: [1..N] real(64);
    var gamma_base: [1..nWake] real(64);
    
    forall elem in 1..N {
        rho_base[elem] = spatialDisc.rhorho_[elem];
        phi_base[elem] = spatialDisc.phi_[elem];
    }
    forall i in 1..nWake {
        gamma_base[i] = spatialDisc.wakeFaceGamma_[i];
    }
    
    // Store base residuals
    spatialDisc.run();
    var res_rho_base: [1..N] real(64);
    var res_phi_base: [1..N] real(64);
    var res_wake_base: [1..nWake] real(64);
    
    forall elem in 1..N {
        res_rho_base[elem] = spatialDisc.res_[elem];
        res_phi_base[elem] = spatialDisc.resPhi_[elem];
    }
    forall i in 1..nWake {
        res_wake_base[i] = spatialDisc.resWake_[i];
    }
    
    // Test a few representative elements
    const testElems = [1, N/4, N/2, 3*N/4, N];
    const testWake = if nWake > 0 then [1, nWake/2, nWake] else [1];
    
    var maxError_dResRho_dRho: real(64) = 0.0;
    var maxError_dResRho_dPhi: real(64) = 0.0;
    var maxError_dResPhi_dRho: real(64) = 0.0;
    var maxError_dResPhi_dPhi: real(64) = 0.0;
    var maxError_dResRho_dGamma: real(64) = 0.0;
    var maxError_dResPhi_dGamma: real(64) = 0.0;
    var maxError_dResWake_dPhi: real(64) = 0.0;
    var maxError_dResWake_dGamma: real(64) = 0.0;
    
    // =========================================================================
    // Test 1: dRes^(rho)/d(rho) - Continuity equation w.r.t. density
    // =========================================================================
    writeln("Testing dRes^(rho)/d(rho)...");
    for testElem in testElems {
        if testElem < 1 || testElem > N then continue;
        
        // Perturb rho at testElem
        spatialDisc.rhorho_[testElem] = rho_base[testElem] + eps;
        spatialDisc.run();
        
        // Compute FD derivatives for all affected rows
        for row in 1..N {
            const fd = (spatialDisc.res_[row] - res_rho_base[row]) / eps;
            const analytical = temporalDisc.A_petsc.get(row-1, testElem-1);
            const error = abs(fd - analytical);
            const relError = if abs(analytical) > 1e-12 then error / abs(analytical) else error;
            
            if error > 1e-6 && relError > 0.01 {
                writeln("  Row ", row, " Col ", testElem, ": FD=", fd, " Analytical=", analytical, " Error=", error, " RelErr=", relError);
            }
            if error > maxError_dResRho_dRho {
                maxError_dResRho_dRho = error;
            }
        }
        
        // Restore
        spatialDisc.rhorho_[testElem] = rho_base[testElem];
    }
    writeln("  Max absolute error: ", maxError_dResRho_dRho);
    
    // =========================================================================
    // Test 2: dRes^(rho)/d(phi) - Continuity equation w.r.t. potential
    // =========================================================================
    writeln("\nTesting dRes^(rho)/d(phi)...");
    for testElem in testElems {
        if testElem < 1 || testElem > N then continue;
        
        // Perturb phi at testElem
        spatialDisc.phi_[testElem] = phi_base[testElem] + eps;
        spatialDisc.run();
        
        // Compute FD derivatives for all affected rows
        for row in 1..N {
            const fd = (spatialDisc.res_[row] - res_rho_base[row]) / eps;
            const analytical = temporalDisc.A_petsc.get(row-1, N + testElem-1);
            const error = abs(fd - analytical);
            const relError = if abs(analytical) > 1e-12 then error / abs(analytical) else error;
            
            if error > 1e-6 && relError > 0.01 {
                writeln("  Row ", row, " Col ", N+testElem, ": FD=", fd, " Analytical=", analytical, " Error=", error, " RelErr=", relError);
            }
            if error > maxError_dResRho_dPhi {
                maxError_dResRho_dPhi = error;
            }
        }
        
        // Restore
        spatialDisc.phi_[testElem] = phi_base[testElem];
    }
    writeln("  Max absolute error: ", maxError_dResRho_dPhi);
    
    // =========================================================================
    // Test 3: dRes^(phi)/d(rho) - Bernoulli equation w.r.t. density
    // =========================================================================
    writeln("\nTesting dRes^(phi)/d(rho)...");
    for testElem in testElems {
        if testElem < 1 || testElem > N then continue;
        
        // Perturb rho at testElem
        spatialDisc.rhorho_[testElem] = rho_base[testElem] + eps;
        spatialDisc.run();
        
        // Compute FD derivatives for all affected rows
        for row in 1..N {
            const fd = (spatialDisc.resPhi_[row] - res_phi_base[row]) / eps;
            const analytical = temporalDisc.A_petsc.get(N + row-1, testElem-1);
            const error = abs(fd - analytical);
            const relError = if abs(analytical) > 1e-12 then error / abs(analytical) else error;
            
            if error > 1e-6 && relError > 0.01 {
                writeln("  Row ", N+row, " Col ", testElem, ": FD=", fd, " Analytical=", analytical, " Error=", error, " RelErr=", relError);
            }
            if error > maxError_dResPhi_dRho {
                maxError_dResPhi_dRho = error;
            }
        }
        
        // Restore
        spatialDisc.rhorho_[testElem] = rho_base[testElem];
    }
    writeln("  Max absolute error: ", maxError_dResPhi_dRho);
    
    // =========================================================================
    // Test 4: dRes^(phi)/d(phi) - Bernoulli equation w.r.t. potential
    // =========================================================================
    writeln("\nTesting dRes^(phi)/d(phi)...");
    for testElem in testElems {
        if testElem < 1 || testElem > N then continue;
        
        // Perturb phi at testElem
        spatialDisc.phi_[testElem] = phi_base[testElem] + eps;
        spatialDisc.run();
        
        // Compute FD derivatives for all affected rows
        for row in 1..N {
            const fd = (spatialDisc.resPhi_[row] - res_phi_base[row]) / eps;
            const analytical = temporalDisc.A_petsc.get(N + row-1, N + testElem-1);
            const error = abs(fd - analytical);
            const relError = if abs(analytical) > 1e-12 then error / abs(analytical) else error;
            
            if error > 1e-6 && relError > 0.01 {
                writeln("  Row ", N+row, " Col ", N+testElem, ": FD=", fd, " Analytical=", analytical, " Error=", error, " RelErr=", relError);
            }
            if error > maxError_dResPhi_dPhi {
                maxError_dResPhi_dPhi = error;
            }
        }
        
        // Restore
        spatialDisc.phi_[testElem] = phi_base[testElem];
    }
    writeln("  Max absolute error: ", maxError_dResPhi_dPhi);
    
    // =========================================================================
    // Test 5: dRes^(rho)/d(Gamma) - Continuity equation w.r.t. circulation
    // =========================================================================
    if nWake > 0 {
        writeln("\nTesting dRes^(rho)/d(Gamma)...");
        for testWakeIdx in testWake {
            if testWakeIdx < 1 || testWakeIdx > nWake then continue;
            
            // Perturb gamma at testWakeIdx
            spatialDisc.wakeFaceGamma_[testWakeIdx] = gamma_base[testWakeIdx] + eps;
            spatialDisc.run();
            
            // Compute FD derivatives for affected rows (wake-adjacent cells)
            const wakeFace = spatialDisc.wakeFace_[testWakeIdx];
            const elem1 = spatialDisc.mesh_.edge2elem_[1, wakeFace];
            const elem2 = spatialDisc.mesh_.edge2elem_[2, wakeFace];
            
            for row in [elem1, elem2] {
                if row > N then continue;
                const fd = (spatialDisc.res_[row] - res_rho_base[row]) / eps;
                const analytical = temporalDisc.A_petsc.get(row-1, 2*N + testWakeIdx-1);
                const error = abs(fd - analytical);
                const relError = if abs(analytical) > 1e-12 then error / abs(analytical) else error;
                
                if error > 1e-6 && relError > 0.01 {
                    writeln("  Row ", row, " Col ", 2*N+testWakeIdx, ": FD=", fd, " Analytical=", analytical, " Error=", error, " RelErr=", relError);
                }
                if error > maxError_dResRho_dGamma {
                    maxError_dResRho_dGamma = error;
                }
            }
            
            // Restore
            spatialDisc.wakeFaceGamma_[testWakeIdx] = gamma_base[testWakeIdx];
        }
        writeln("  Max absolute error: ", maxError_dResRho_dGamma);
    }
    
    // =========================================================================
    // Test 6: dRes^(phi)/d(Gamma) - Bernoulli equation w.r.t. circulation
    // =========================================================================
    if nWake > 0 {
        writeln("\nTesting dRes^(phi)/d(Gamma)...");
        for testWakeIdx in testWake {
            if testWakeIdx < 1 || testWakeIdx > nWake then continue;
            
            // Perturb gamma at testWakeIdx
            spatialDisc.wakeFaceGamma_[testWakeIdx] = gamma_base[testWakeIdx] + eps;
            spatialDisc.run();
            
            // Compute FD derivatives for affected rows (wake-adjacent cells)
            const wakeFace = spatialDisc.wakeFace_[testWakeIdx];
            const elem1 = spatialDisc.mesh_.edge2elem_[1, wakeFace];
            const elem2 = spatialDisc.mesh_.edge2elem_[2, wakeFace];
            
            for row in [elem1, elem2] {
                if row > N then continue;
                const fd = (spatialDisc.resPhi_[row] - res_phi_base[row]) / eps;
                const analytical = temporalDisc.A_petsc.get(N + row-1, 2*N + testWakeIdx-1);
                const error = abs(fd - analytical);
                const relError = if abs(analytical) > 1e-12 then error / abs(analytical) else error;
                
                if error > 1e-6 && relError > 0.01 {
                    writeln("  Row ", N+row, " Col ", 2*N+testWakeIdx, ": FD=", fd, " Analytical=", analytical, " Error=", error, " RelErr=", relError);
                }
                if error > maxError_dResPhi_dGamma {
                    maxError_dResPhi_dGamma = error;
                }
            }
            
            // Restore
            spatialDisc.wakeFaceGamma_[testWakeIdx] = gamma_base[testWakeIdx];
        }
        writeln("  Max absolute error: ", maxError_dResPhi_dGamma);
    }
    
    // =========================================================================
    // Test 7: dRes^(wake)/d(phi) - Wake equation w.r.t. potential
    // =========================================================================
    if nWake > 0 {
        writeln("\nTesting dRes^(wake)/d(phi)...");
        for testElem in testElems {
            if testElem < 1 || testElem > N then continue;
            
            // Perturb phi at testElem
            spatialDisc.phi_[testElem] = phi_base[testElem] + eps;
            spatialDisc.run();
            
            // Compute FD derivatives for wake rows
            for wakeIdx in 1..nWake {
                const fd = (spatialDisc.resWake_[wakeIdx] - res_wake_base[wakeIdx]) / eps;
                const analytical = temporalDisc.A_petsc.get(2*N + wakeIdx-1, N + testElem-1);
                const error = abs(fd - analytical);
                const relError = if abs(analytical) > 1e-12 then error / abs(analytical) else error;
                
                if error > 1e-6 && relError > 0.01 {
                    writeln("  Row ", 2*N+wakeIdx, " Col ", N+testElem, ": FD=", fd, " Analytical=", analytical, " Error=", error, " RelErr=", relError);
                }
                if error > maxError_dResWake_dPhi {
                    maxError_dResWake_dPhi = error;
                }
            }
            
            // Restore
            spatialDisc.phi_[testElem] = phi_base[testElem];
        }
        writeln("  Max absolute error: ", maxError_dResWake_dPhi);
    }
    
    // =========================================================================
    // Test 8: dRes^(wake)/d(Gamma) - Wake equation w.r.t. circulation
    // =========================================================================
    if nWake > 0 {
        writeln("\nTesting dRes^(wake)/d(Gamma)...");
        for testWakeIdx in testWake {
            if testWakeIdx < 1 || testWakeIdx > nWake then continue;
            
            // Perturb gamma at testWakeIdx
            spatialDisc.wakeFaceGamma_[testWakeIdx] = gamma_base[testWakeIdx] + eps;
            spatialDisc.run();
            
            // Compute FD derivatives for wake rows
            for wakeIdx in 1..nWake {
                const fd = (spatialDisc.resWake_[wakeIdx] - res_wake_base[wakeIdx]) / eps;
                const analytical = temporalDisc.A_petsc.get(2*N + wakeIdx-1, 2*N + testWakeIdx-1);
                const error = abs(fd - analytical);
                const relError = if abs(analytical) > 1e-12 then error / abs(analytical) else error;
                
                if error > 1e-6 && relError > 0.01 {
                    writeln("  Row ", 2*N+wakeIdx, " Col ", 2*N+testWakeIdx, ": FD=", fd, " Analytical=", analytical, " Error=", error, " RelErr=", relError);
                }
                if error > maxError_dResWake_dGamma {
                    maxError_dResWake_dGamma = error;
                }
            }
            
            // Restore
            spatialDisc.wakeFaceGamma_[testWakeIdx] = gamma_base[testWakeIdx];
        }
        writeln("  Max absolute error: ", maxError_dResWake_dGamma);
    }
    
    // =========================================================================
    // Summary
    // =========================================================================
    writeln("\n=== JACOBIAN VERIFICATION SUMMARY ===");
    writeln("Block                  | Max Absolute Error");
    writeln("-----------------------|-------------------");
    writeln("dRes^(rho)/d(rho)      | ", maxError_dResRho_dRho);
    writeln("dRes^(rho)/d(phi)      | ", maxError_dResRho_dPhi);
    writeln("dRes^(phi)/d(rho)      | ", maxError_dResPhi_dRho);
    writeln("dRes^(phi)/d(phi)      | ", maxError_dResPhi_dPhi);
    if nWake > 0 {
        writeln("dRes^(rho)/d(Gamma)    | ", maxError_dResRho_dGamma);
        writeln("dRes^(phi)/d(Gamma)    | ", maxError_dResPhi_dGamma);
        writeln("dRes^(wake)/d(phi)     | ", maxError_dResWake_dPhi);
        writeln("dRes^(wake)/d(Gamma)   | ", maxError_dResWake_dGamma);
    }
    writeln("");
    
    // Restore state
    forall elem in 1..N {
        spatialDisc.rhorho_[elem] = rho_base[elem];
        spatialDisc.phi_[elem] = phi_base[elem];
    }
    forall i in 1..nWake {
        spatialDisc.wakeFaceGamma_[i] = gamma_base[i];
    }
    spatialDisc.run();
}

// Main entry point for testing
proc runJacobianTest() {
    writeln("=== Jacobian Verification Test ===\n");
    
    var inputs = new potentialInputs();
    inputs.initializeFlowField();
    
    var Mesh = new shared MeshData(inputs.GRID_FILENAME_, inputs.ELEMENT_TYPE_);
    Mesh.buildConnectivity();
    
    var spatialDisc = new shared unsteadySpatialDiscretization(Mesh, inputs);
    spatialDisc.initializeMetrics();
    spatialDisc.initializeKuttaCells();
    spatialDisc.initializeSolution();
    spatialDisc.run();
    
    var temporalDisc = new shared unsteadyTemporalDiscretization(spatialDisc, inputs);
    temporalDisc.initialize();
    
    verifyJacobian(temporalDisc);
}

}

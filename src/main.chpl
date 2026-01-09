use IO;
use Time;
use Math;
import input.potentialInputs;
use mesh;
use spatialDiscretization;
use temporalDiscretization;
use unsteadySpatialDiscretization;
use unsteadyTemporalDiscretization;
use testMetrics;
use linearAlgebra;

config const runTests = false;  // Use --runTests=true to enable verification

proc main() {
    
    if runTests {
        run_tests();
    }
    else {
        writeln("Full Potential Solver - Chapel Implmentation");
        var t_ini: stopwatch;
        t_ini.start();

        var inputs = new potentialInputs();
        inputs.initializeFlowField();
        
        // =====================================================================
        // Mach Continuation: solve at progressively higher Mach numbers
        // =====================================================================
        if inputs.MACH_CONTINUATION_ {
            var currentMach = inputs.MACH_START_;
            const targetMach = inputs.MACH_TARGET_;
            const machStep = inputs.MACH_STEP_;
            var totalIterations = 0;
            
            writeln("=======================================================");
            writeln("  MACH CONTINUATION: ", currentMach, " -> ", targetMach);
            writeln("=======================================================");
            
            // Set initial Mach BEFORE creating discretization objects
            inputs.setMach(currentMach);
            
            var Mesh = new shared MeshData(inputs.GRID_FILENAME_, inputs.ELEMENT_TYPE_);
            Mesh.buildConnectivity();
            var spatialDisc = new shared spatialDiscretization(Mesh, inputs);
            var steadySolver = new shared temporalDiscretization(spatialDisc, inputs);
            
            steadySolver.initialize();
            
            while currentMach <= targetMach + 1e-10 {
                writeln("\n>>> Solving at Mach = ", currentMach, " <<<");
                
                steadySolver.solve();
                totalIterations += steadySolver.it_;
                
                // Move to next Mach number
                currentMach += machStep;
                
                if currentMach <= targetMach + 1e-10 {
                    // Clamp to target to avoid overshooting
                    if currentMach > targetMach {
                        currentMach = targetMach;
                    }
                    inputs.setMach(currentMach);
                    steadySolver.resetForContinuation(inputs);
                }
            }
            
            writeln("\n=======================================================");
            writeln("  MACH CONTINUATION COMPLETE");
            writeln("  Total iterations across all Mach steps: ", totalIterations);
            writeln("=======================================================");
        }
        else {
            // Standard single-Mach solve
            var Mesh = new shared MeshData(inputs.GRID_FILENAME_, inputs.ELEMENT_TYPE_);
            Mesh.buildConnectivity();
            if inputs.FLOW_ == "steady" {
                var spatialDisc = new shared spatialDiscretization(Mesh, inputs);
                var steadySolver = new shared temporalDiscretization(spatialDisc, inputs);
                
                steadySolver.initialize();
                steadySolver.solve();
            }
            else if inputs.FLOW_ == "unsteady" {
                var spatialDisc = new shared unsteadySpatialDiscretization(Mesh, inputs);
                spatialDisc.initializeMetrics();
                spatialDisc.initializeKuttaCells();
                spatialDisc.initializeSolution();
                // spatialDisc.run();
                // spatialDisc.writeSolution();

                // const res = RMSE(spatialDisc.res_, spatialDisc.elemVolume_);
                // const resPhi = RMSE(spatialDisc.resPhi_, spatialDisc.elemVolume_);
                // const resWake = RMSE(spatialDisc.resWake_);

                // writeln("Initial Residuals - Total: ", res, ", Phi: ", resPhi, ", Wake: ", resWake);

                var unsteadySolver = new shared unsteadyTemporalDiscretization(spatialDisc, inputs);
                unsteadySolver.initialize();
                unsteadySolver.solve();
            }
        }

        t_ini.stop();
        writeln("total execution : ", t_ini.elapsed(), " seconds");
    }
    
}
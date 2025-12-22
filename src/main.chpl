use IO;
use Time;
use Math;
import input.potentialInputs;
use mesh;
use spatialDiscretization;
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
        var Mesh = new shared MeshData(inputs.GRID_FILENAME_, inputs.ELEMENT_TYPE_);
        Mesh.buildConnectivity();

        var spatialDisc = new shared spatialDiscretization(Mesh, inputs);
        spatialDisc.initializeMetrics();
        spatialDisc.initializeSolution();
        spatialDisc.run();


        
        spatialDisc.writeSolution();

        t_ini.stop();
        writeln("total execution : ", t_ini.elapsed(), " seconds");
    }
    
}
use IO;
use Time;
import input.potentialInputs;
use mesh;
use spatialDiscretization;
use testMetrics;

config const runTests = false;  // Use --runTests=true to enable metric verification

proc main() {
    writeln("Full Potential Solver - Chapel Implmentation");
    var t_ini: stopwatch;
    t_ini.start();

    var inputs = new potentialInputs();
    var Mesh = new shared MeshData(inputs.GRID_FILENAME_, inputs.ELEMENT_TYPE_);
    Mesh.buildConnectivity();

    var spatialDisc = new shared spatialDiscretization(Mesh, inputs);
    spatialDisc.initializeMetrics();
    
    spatialDisc.initializeSolution();
    // spatialDisc.updateGhostCells();
    spatialDisc.computeVelocityFromPhi();
    spatialDisc.writeSolution();

    t_ini.stop();
    writeln("total execution : ", t_ini.elapsed(), " seconds");
    
    if runTests {
        verifyMetrics(spatialDisc);
        spatialDisc.debugGradientAtElement(1024, spatialDisc.phi_);
    }
}
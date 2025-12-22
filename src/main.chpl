use IO;
use Time;
use Math;
import input.potentialInputs;
use mesh;
use spatialDiscretization;
use testMetrics;
use linearAlgebra;

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

    t_ini.stop();
    writeln("total execution : ", t_ini.elapsed(), " seconds");
    
    if runTests {
        verifyMetrics(spatialDisc);
        spatialDisc.debugGradientAtElement(1024, spatialDisc.phi_);
    }
}
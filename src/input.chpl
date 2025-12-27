use Math;

config const GRID_FILENAME : string;
config const START_FILENAME : string;

config const FREEZE_CIRCULATION : bool = false; // If true, the circulation is frozen at the initial value
config const WAKE_CONV_TOL : real(64);
config const CIRCULATION_CONV_TOL : real(64);

config const ELEMENT_TYPE : string = "QuadElements";
config const OUTPUT_FILENAME : string = "output.cgns";
config const CGNS_OUTPUT_FREQ : int;
config const MACH : real(64);
config const ALPHA : real(64);
config const TEMPERATURE : real(64);
config const GAMMA : real(64);
config const CFL : real(64);
config const CFL_RAMP_FACTOR : real(64);
config const CFL_RAMP_IT : int;
config const CFL_RAMP_MAX : real(64);
config const OMEGA_CIRCULATION : real(64); // Relaxation factor for circulation update
config const OMEGA : real(64); // Relaxation factor for the iterative solver
config const IT_MAX : int;
config const CONV_TOL : real(64);
config const X_REF : real(64);
config const Y_REF : real(64);
config const MU_C : real(64);
config const MACH_C : real(64);
config const BETA : real(64); // Level of upwinding in Jacobian matrix

config const GMRES_RTOL : real(64);
config const GMRES_ATOL : real(64);
config const GMRES_DTOL : real(64);
config const GMRES_MAXIT : int;
config const GMRES_RESTART : int;
config const GMRES_PRECON : string;

config const DUAL_TIME_STEP : bool;
config const ALPHA_0 : real(64);
config const ALPHA_AMPLITUDE : real(64);
config const ALPHA_FREQUENCY : real(64);
config const ALPHA_PHASE : real(64);
config const TIME_STEP : real(64);
config const TIME_FINAL : real(64);
config const CFL_UNSTEADY : real(64);

config const q_crit : real(64);

record potentialInputs {
    var GRID_FILENAME_: string = GRID_FILENAME;
    var START_FILENAME_: string = START_FILENAME;
    var FREEZE_CIRCULATION_: bool = FREEZE_CIRCULATION;
    var WAKE_CONV_TOL_: real(64) = WAKE_CONV_TOL;
    var CIRCULATION_CONV_TOL_: real(64) = CIRCULATION_CONV_TOL;
    var ELEMENT_TYPE_: string = ELEMENT_TYPE;
    var OUTPUT_FILENAME_: string = OUTPUT_FILENAME;
    var CGNS_OUTPUT_FREQ_: int = CGNS_OUTPUT_FREQ;
    var MACH_: real(64) = MACH;
    var ALPHA_: real(64) = ALPHA;
    var TEMPERATURE_: real(64) = TEMPERATURE;   
    var GAMMA_: real(64) = GAMMA;
    var CFL_ : real(64) = CFL;
    var CFL_RAMP_FACTOR_ : real(64) = CFL_RAMP_FACTOR;
    var CFL_RAMP_IT_ : int = CFL_RAMP_IT;
    var CFL_RAMP_MAX_ : real(64) = CFL_RAMP_MAX;
    var OMEGA_CIRCULATION_ : real(64) = OMEGA_CIRCULATION;
    var OMEGA_ : real(64) = OMEGA;
    var IT_MAX_: int = IT_MAX;
    var CONV_TOL_ : real(64) = CONV_TOL;
    
    var RHO_INF_: real(64) = 1.0;
    var VEL_INF_: real(64) = 1.0;
    var a_INF_ = VEL_INF_ / MACH_;
    var U_INF_ = VEL_INF_ * cos(ALPHA_ * pi / 180.0);
    var V_INF_ = VEL_INF_ * sin(ALPHA_ * pi / 180.0);
    var P_INF_ = RHO_INF_ ** GAMMA_ / (GAMMA_ * MACH_ * MACH_);
    var Q_INF_ = 0.5 * RHO_INF_ * VEL_INF_ * VEL_INF_;
    
    var X_REF_: real(64) = X_REF;
    var Y_REF_: real(64) = Y_REF;
    var S_REF_: real(64) = 1.0; // Reference length for non-dimensionalization
    
    var MU_C_: real(64) = MU_C;
    var MACH_C_: real(64) = MACH_C;
    var BETA_: real(64) = BETA;

    var GMRES_RTOL_: real(64) = GMRES_RTOL;
    var GMRES_ATOL_: real(64) = GMRES_ATOL;
    var GMRES_DTOL_: real(64) = GMRES_DTOL;
    var GMRES_MAXIT_: int = GMRES_MAXIT;
    var GMRES_RESTART_: int = GMRES_RESTART;
    var GMRES_PRECON_: string = GMRES_PRECON;

    var DUAL_TIME_STEP_: bool = DUAL_TIME_STEP;
    var ALPHA_0_: real(64) = ALPHA_0;
    var ALPHA_AMPLITUDE_: real(64) = ALPHA_AMPLITUDE;
    var ALPHA_FREQUENCY_: real(64) = ALPHA_FREQUENCY;
    var ALPHA_PHASE_: real(64) = ALPHA_PHASE;
    var TIME_STEP_: real(64) = TIME_STEP;
    var TIME_FINAL_: real(64) = TIME_FINAL;
    var CFL_UNSTEADY_: real(64) = CFL_UNSTEADY;

    var q_crit_: real(64) = q_crit;
    var beta_crit_ = MACH_**2*q_crit / (1 + (GAMMA_ - 1)/2*MACH_**2*(1 - q_crit**2));
    var rho_crit_ = (1.0 + (GAMMA_ - 1)/2 * MACH_ * MACH_ * (1.0 - q_crit * q_crit)) ** (1.0 / (GAMMA_ - 1.0));

    proc init() {
        writeln("GRID_FILENAME = ", GRID_FILENAME);
        writeln("START_FILENAME = ", START_FILENAME);
        writeln("ELEMENT_TYPE = ", ELEMENT_TYPE);
        writeln("OUTPUT_FILENAME = ", OUTPUT_FILENAME);
        writeln("MACH = ", MACH);
        writeln("ALPHA = ", ALPHA);
        writeln("TEMPERATURE = ", TEMPERATURE);
        writeln("GAMMA = ", GAMMA);
        writeln("CFL = ", CFL);
        writeln("IT_MAX = ", IT_MAX);
        writeln("X_REF = ", X_REF);
        writeln("Y_REF = ", Y_REF);
        writeln("MU_C = ", MU_C);
        writeln("MACH_C = ", MACH_C);
    }
}

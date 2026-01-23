use Math;
config const FLOW : string = "steady"; // "steady" or "unsteady"

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
config const CONV_ATOL : real(64) = 1e-15;  // Absolute convergence tolerance for residual
config const FREEZE_MU_TOL : real(64) = 1e-6; // Residual tolerance to freeze MU adaptation
config const X_REF : real(64);
config const Y_REF : real(64);
config const MU_C : real(64);
config const MACH_C : real(64);
config const BETA : real(64); // Level of upwinding in Jacobian matrix

// Adaptive upwinding parameters (Crovato thesis)
config const ADAPTIVE_UPWIND : bool = false;  // Enable adaptive MU_C/MACH_C
config const MU_C_START : real(64) = 2.0;     // Initial MU_C (strong stabilization)
config const MU_C_FINAL : real(64) = 1.0;     // Final MU_C (accurate)
config const MACH_C_START : real(64) = 0.92;  // Initial MACH_C (wide supersonic region)
config const MACH_C_FINAL : real(64) = 0.95;  // Final MACH_C (narrow supersonic region)
config const ADAPT_THRESHOLD : real(64) = 1e-2;  // Residual threshold to switch parameters

// Adaptive BETA for Newton consistency (reduce BETA near convergence for quadratic rate)
config const ADAPTIVE_BETA : bool = false;    // Enable adaptive BETA reduction
config const BETA_START : real(64) = 5e-3;    // Initial BETA (strong Jacobian conditioning)
config const BETA_FINAL : real(64) = 0.0;     // Final BETA (pure Newton for quadratic convergence)
config const BETA_THRESHOLD : real(64) = 1e-4; // Residual threshold to reduce BETA

config const LINE_SEARCH : bool = false;
config const MAX_LINE_SEARCH : int = 5;
config const SUFFICIENT_DECREASE : real(64) = 1.2;

// Selective Frequency Damping (SFD) parameters (Jordi et al. 2014 encapsulated formulation)
config const SFD_ENABLED : bool = false;     // Enable SFD acceleration
config const SFD_CHI : real(64) = 0.5;       // Damping coefficient χ (0.1-1.0 typical)
config const SFD_DELTA : real(64) = 0.1;     // Filter width Δ = ωc·dt (0.01-0.5 typical)

// Mach continuation parameters
config const MACH_CONTINUATION : bool = false;  // Enable Mach continuation
config const MACH_START : real(64) = 0.5;       // Starting Mach number for continuation
config const MACH_STEP : real(64) = 0.05;       // Mach number increment per step

config const GMRES_RTOL : real(64);
config const GMRES_ATOL : real(64);
config const GMRES_DTOL : real(64);
config const GMRES_MAXIT : int;
config const GMRES_RESTART : int;
config const GMRES_PRECON : string;
config const GMRES_PRECON_SIDE : string = "right";  // Preconditioning side: "left" or "right"
config const JACOBIAN_TYPE : string = "analytical";  // Jacobian type: "analytical" or "numerical"
config const USE_NATIVE_GMRES : bool = false;  // Use Chapel-native GMRES instead of PETSc

// Adaptive (Inexact) Newton parameters - Eisenstat-Walker forcing terms
config const ADAPTIVE_GMRES_TOL : bool = false;  // Enable adaptive GMRES tolerance
config const GMRES_RTOL_MAX : real(64) = 0.1;    // Max (loose) tolerance for early iterations
config const GMRES_RTOL_MIN : real(64) = 1e-4;  // Min (tight) tolerance for late iterations
config const GMRES_RTOL_ETA : real(64) = 0.9;   // Eisenstat-Walker eta parameter (0 < eta < 1)


config const DUAL_TIME_STEP : bool;
config const ALPHA_0 : real(64);
config const ALPHA_AMPLITUDE : real(64);
config const ALPHA_FREQUENCY : real(64);
config const ALPHA_PHASE : real(64);
config const TIME_STEP : real(64);
config const TIME_FINAL : real(64);
config const CFL_UNSTEADY : real(64);
config const WAKE_CONVECTION_VELOCITY : real(64) = 1.0;  // Wake convection velocity (default: freestream)
config const LAGRANGIAN_WAKE : bool = true;  // Use Lagrangian vortex shedding (default: true for unsteady)

config const q_crit : real(64);

// Cylinder analytical farfield BC
config const FARFIELD_BC_TYPE : string = "freestream";  // "freestream" or "cylinder"
config const CYLINDER_RADIUS : real(64) = 0.5;  // Cylinder radius for analytical BC

record potentialInputs {
    var FLOW_ : string = FLOW;
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
    var CONV_ATOL_ : real(64) = CONV_ATOL;  // Absolute convergence tolerance
    var FREEZE_MU_TOL_ : real(64) = FREEZE_MU_TOL; // Residual tolerance to freeze MU adaptation
    
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

    // Adaptive upwinding (Crovato thesis)
    var ADAPTIVE_UPWIND_: bool = ADAPTIVE_UPWIND;
    var MU_C_START_: real(64) = MU_C_START;
    var MU_C_FINAL_: real(64) = MU_C_FINAL;
    var MACH_C_START_: real(64) = MACH_C_START;
    var MACH_C_FINAL_: real(64) = MACH_C_FINAL;
    var ADAPT_THRESHOLD_: real(64) = ADAPT_THRESHOLD;
    var upwindAdapted_: bool = false;  // Track if we've switched to final values

    // Adaptive BETA for Newton consistency
    var ADAPTIVE_BETA_: bool = ADAPTIVE_BETA;
    var BETA_START_: real(64) = BETA_START;
    var BETA_FINAL_: real(64) = BETA_FINAL;
    var BETA_THRESHOLD_: real(64) = BETA_THRESHOLD;
    var betaAdapted_: bool = false;  // Track if we've switched to final BETA

    var LINE_SEARCH_: bool = LINE_SEARCH;
    var MAX_LINE_SEARCH_: int = MAX_LINE_SEARCH;
    var SUFFICIENT_DECREASE_: real(64) = SUFFICIENT_DECREASE;

    // SFD parameters
    var SFD_ENABLED_: bool = SFD_ENABLED;
    var SFD_CHI_: real(64) = SFD_CHI;
    var SFD_DELTA_: real(64) = SFD_DELTA;

    // Mach continuation parameters
    var MACH_CONTINUATION_: bool = MACH_CONTINUATION;
    var MACH_START_: real(64) = MACH_START;
    var MACH_STEP_: real(64) = MACH_STEP;
    var MACH_TARGET_: real(64) = MACH;  // Store the target Mach number

    var GMRES_RTOL_: real(64) = GMRES_RTOL;
    var GMRES_ATOL_: real(64) = GMRES_ATOL;
    var GMRES_DTOL_: real(64) = GMRES_DTOL;
    var GMRES_MAXIT_: int = GMRES_MAXIT;
    var GMRES_RESTART_: int = GMRES_RESTART;
    var GMRES_PRECON_: string = GMRES_PRECON;
    var GMRES_PRECON_SIDE_: string = GMRES_PRECON_SIDE;
    var JACOBIAN_TYPE_: string = JACOBIAN_TYPE;
    var USE_NATIVE_GMRES_: bool = USE_NATIVE_GMRES;

    // Adaptive (Inexact) Newton parameters
    var ADAPTIVE_GMRES_TOL_: bool = ADAPTIVE_GMRES_TOL;
    var GMRES_RTOL_MAX_: real(64) = GMRES_RTOL_MAX;
    var GMRES_RTOL_MIN_: real(64) = GMRES_RTOL_MIN;
    var GMRES_RTOL_ETA_: real(64) = GMRES_RTOL_ETA;

    var DUAL_TIME_STEP_: bool = DUAL_TIME_STEP;
    var ALPHA_0_: real(64) = ALPHA_0;
    var ALPHA_AMPLITUDE_: real(64) = ALPHA_AMPLITUDE;
    var ALPHA_FREQUENCY_: real(64) = ALPHA_FREQUENCY;
    var ALPHA_PHASE_: real(64) = ALPHA_PHASE;
    var TIME_STEP_: real(64) = TIME_STEP;
    var TIME_FINAL_: real(64) = TIME_FINAL;
    var CFL_UNSTEADY_: real(64) = CFL_UNSTEADY;
    var WAKE_CONVECTION_VELOCITY_: real(64) = WAKE_CONVECTION_VELOCITY;  // Wake convection velocity
    var LAGRANGIAN_WAKE_: bool = LAGRANGIAN_WAKE;  // Use Lagrangian vortex shedding

    var q_crit_: real(64) = q_crit;
    var beta_crit_ = MACH_**2*q_crit / (1 + (GAMMA_ - 1)/2*MACH_**2*(1 - q_crit**2));
    var rho_crit_ = (1.0 + (GAMMA_ - 1)/2 * MACH_ * MACH_ * (1.0 - q_crit * q_crit)) ** (1.0 / (GAMMA_ - 1.0));

    // Cylinder analytical farfield BC
    var FARFIELD_BC_TYPE_: string = FARFIELD_BC_TYPE;
    var CYLINDER_RADIUS_: real(64) = CYLINDER_RADIUS;

    proc init() {
        writeln("Flow type: ", FLOW);
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
        if MACH_CONTINUATION {
            writeln("MACH_CONTINUATION enabled: ", MACH_START, " -> ", MACH, " (step ", MACH_STEP, ")");
        }
    }

    proc ref initializeFlowField() {
        if this.FLOW_ == "steady"{
            this.RHO_INF_ = 1.0;
            this.VEL_INF_ = 1.0;
            this.a_INF_ = this.VEL_INF_ / this.MACH_;
            this.U_INF_ = this.VEL_INF_ * cos(this.ALPHA_ * pi / 180.0);
            this.V_INF_ = this.VEL_INF_ * sin(this.ALPHA_ * pi / 180.0);
            this.P_INF_ = this.RHO_INF_ ** this.GAMMA_ / (this.GAMMA_ * this.MACH_ * this.MACH_);
            this.Q_INF_ = 0.5 * this.RHO_INF_ * this.VEL_INF_ * this.VEL_INF_;
        }
        // else if this.FLOW_ == "unsteady" {
        //     this.RHO_INF_ = 1.0;
        //     this.a_INF_ = 1.0;
        //     this.VEL_INF_ = this.MACH_ * this.a_INF_;
        //     this.U_INF_ = this.VEL_INF_ * cos(this.ALPHA_ * pi / 180.0);
        //     this.V_INF_ = this.VEL_INF_ * sin(this.ALPHA_ * pi / 180.0);
        //     this.P_INF_ = this.RHO_INF_ * this.a_INF_**2 / this.GAMMA_;
        //     this.Q_INF_ = 0.5 * this.RHO_INF_ * this.VEL_INF_ * this.VEL_INF_;
        // }
    }

    // Update Mach number and recompute derived quantities
    proc ref setMach(newMach: real(64)) {
        this.MACH_ = newMach;
        this.a_INF_ = this.VEL_INF_ / this.MACH_;
        this.P_INF_ = this.RHO_INF_ ** this.GAMMA_ / (this.GAMMA_ * this.MACH_ * this.MACH_);
        this.beta_crit_ = this.MACH_**2 * this.q_crit_ / (1 + (this.GAMMA_ - 1)/2 * this.MACH_**2 * (1 - this.q_crit_**2));
        this.rho_crit_ = (1.0 + (this.GAMMA_ - 1)/2 * this.MACH_ * this.MACH_ * (1.0 - this.q_crit_ * this.q_crit_)) ** (1.0 / (this.GAMMA_ - 1.0));
    }

    // Update angle of attack and recompute velocity components
    proc ref setAlpha(newAlpha: real(64)) {
        this.ALPHA_ = newAlpha;
        this.U_INF_ = this.VEL_INF_ * cos(this.ALPHA_ * pi / 180.0);
        this.V_INF_ = this.VEL_INF_ * sin(this.ALPHA_ * pi / 180.0);
    }

    // Compute instantaneous angle of attack for oscillating airfoil
    // α(t) = α_0 + Δα * sin(2π * f * t + φ)
    proc getOscillatingAlpha(t: real(64)): real(64) {
        return this.ALPHA_0_ + this.ALPHA_AMPLITUDE_ * sin(this.ALPHA_FREQUENCY_ * t + this.ALPHA_PHASE_);
    }

    // Get effective time step for temporal discretization
    // For dual time stepping (oscillating airfoil): use physical TIME_STEP
    // For pseudo time stepping: use CFL
    proc getTimeStep(): real(64) {
        if this.DUAL_TIME_STEP_ {
            return this.TIME_STEP_;
        } else {
            return this.CFL_;
        }
    }
}

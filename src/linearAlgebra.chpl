module linearAlgebra
{
use LinearAlgebra;
use List;
use PETSCapi;
use C_PETSC;
use petsc;
use CTypes;

proc l2Norm(array : [] real(64)) {
    var norm: real(64) = 0.0;
    forall val in array with (+ reduce norm) {
        norm += val*val;
    }

    return sqrt(norm);
}

proc RMSE(array : [] real(64), volumeArr : [] real(64)) {
    var norm: real(64) = 0.0;
    forall (val, vol) in zip(array, volumeArr) with (+ reduce norm) {
        norm += (val / vol) * (val * vol);
    }

    return sqrt(norm / array.size);
}

proc RMSE(array : [] real(64)) {
    var norm: real(64) = 0.0;
    forall val in array with (+ reduce norm) {
        norm += val * val;
    }

    return sqrt(norm / array.size);
}

proc GMRES(ref ksp: PETSCksp_c, const ref A: PETSCmatrix_c, const ref b: PETSCvector_c, ref x: PETSCvector_c) {
    ksp.setOperators(A, A);
    ksp.solve(b, x);

    var its = ksp.getIterationNumber();
    var reason = ksp.getConvergedReason();

    // if reason < 0 {
    //     writeln("GMRES did not converge, reason: ", reason);
    // }
    return (its, reason);

}

proc GMRES(const ref A: [], const ref b: [], ref x: [], itmax = 20) {
    var M = A.dim(0).size;
    var N = A.dim(1).size;

    var A_petsc : PETSCmatrix_c =  new owned PETSCmatrix_c(PETSC_COMM_SELF, "seqaij", M, M, N, N);
    var nnz : [0..M-1] PetscInt;
    nnz = N;
    A_petsc.preAllocate(nnz);
    for (i, j) in A.domain {
        if A[i, j] != 0.0 {
            // writeln("i = ", i, ", j = ", j, " A[", i, ", ", j, "] = ", A[i, j]);
            A_petsc.set(i, j, A[i, j]);
        }
    }
    A_petsc.assemblyComplete();
    // A_petsc.matView();

    var b_petc : PETSCvector_c = new owned PETSCvector_c(PETSC_COMM_SELF, N, N, 0.0, "seq");
    b_petc.set(b);
    b_petc.assemblyComplete();
    // A_petsc.vecView(b_petc);

    var x_petc : PETSCvector_c = new owned PETSCvector_c(PETSC_COMM_SELF, N, N, 0.0, "seq");
    x_petc.set(x);
    x_petc.assemblyComplete();
    // A_petsc.vecView(x_petc);

    var ksp = new PETSCksp_c(PETSC_COMM_SELF, "gmres");
    ksp.setOperators(A_petsc, A_petsc);
    ksp.setTolerances(1e-6, 1e-12, 1e5, 1000);
    ksp.GMRESSetRestart(30);
    ksp.GMRESSetPreAllocateVectors();
    ksp.solve(b_petc, x_petc);

    var its = ksp.getIterationNumber();
    var reason = ksp.getConvergedReason();

    for i in x.domain {
        x[i] = x_petc(i);
    }

    if reason < 0 {
        writeln("GMRES did not converge, reason: ", reason);
    }

}

proc gaussSeidel(const ref A: [], const ref b: [], ref x: [], itmax = 20) {
    const dom = x.domain;
    const n = dom.high;
    var error = 100.0;
    var it = 1;

    while error > 1e-12 {
        // Gauss-Seidel update step
        for i in dom {
            var sum1 = 0.0;
            for j in dom.low..i-1 {
                sum1 += A[i,j]*x[j];   // updated x
            }

            var sum2 = 0.0;
            for j in i+1..dom.high {
                sum2 += A[i,j]*x[j];   // still old x
            }

            x[i] = (b[i] - sum1 - sum2) / A[i,i];
        }

        // Compute residual r = b - A*x
        var r: [dom] real(64);
        forall i in dom {
            var Ax_i = 0.0;
            for j in dom {
                Ax_i += A[i,j] * x[j];
            }
            r[i] = b[i] - Ax_i;
        }

        error = l2Norm(r);
        it += 1;

        // writeln("Iteration ", it, ", Error = ", error);

        if it >= itmax {
            // writeln("Gauss-Seidel converged after ", it, " iterations with error ", error);
            break;
        }
    }
}

proc gaussSeidelSparse(const ref A: [] real(64), const ref b: [] real(64), ref x: [] real(64), itmax = 20) {
    const dom = x.domain;
    var error = 100.0;
    var it = 1;

    // Extract row-wise non-zero structure for easier access
    var rowMap: [dom] list(int);
    for (i, j) in A.domain {
        rowMap[i].pushBack(j);
    }

    while error > 1e-12 {
        for i in dom {
            var sigma = 0.0;
            var diag = 0.0;

            for j in rowMap[i] {
                if j == i {
                    diag = A[i, j]; // Save diagonal
                } 
                else {
                    sigma += A[i, j] * x[j]; // Off-diagonal
                }
            }

            x[i] = (b[i] - sigma) / diag;
        }

        // Compute residual r = b - A*x
        var r: [dom] real(64);
        forall i in dom {
            var Ax_i = 0.0;
            for j in rowMap[i] {
                Ax_i += A[i, j] * x[j];
            }
            r[i] = b[i] - Ax_i;
        }

        error = l2Norm(r);

        if it >= itmax then break;
        it += 1;
    }
}

proc jacobiSparse(const ref A: [] real(64), const ref b: [] real(64), ref x: [] real(64), itmax = 10) {
    const dom = x.domain;
    var error = 100.0;
    var it = 1;

    // Build row-wise non-zero structure for easier access
    var rowMap: [dom] list(int);
    for (i, j) in A.domain {
        rowMap[i].pushBack(j);
    }

    var x_new: [dom] real(64);

    while error > 1e-20 {
        // Compute new iterate in parallel using 'forall'
        forall i in dom {
            var sigma = 0.0;
            var diag = 0.0;
            for j in rowMap[i] {
                    if j == i then
                    diag = A[i,j];
                    else
                    sigma += A[i,j] * x[j];
            }
            x_new[i] = (b[i] - sigma) / diag;
        }

        // // Compute residual r = b - A*x_new in parallel
        // var r: [dom] real(64);
        // forall i in dom {
        // var Ax_i = 0.0;
        // for j in rowMap[i] {
        //     Ax_i += A[i,j] * x_new[j];
        // }
        // r[i] = b[i] - Ax_i;
        // }
        // error = l2Norm(r);

        x = x_new;
        if it >= itmax then break;
        it += 1;
    }
}

}
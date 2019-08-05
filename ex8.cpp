#include "tensors.hpp"
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <stdlib.h> 




class matrixKron
{
public:
	static PetscErrorCode matMult(Mat dummy, Vec source, Vec target);
	Mat A;
	Mat B;

	int arows;
	int acols;
	int brows;
	int bcols;

	PetscErrorCode setupAB(int m, int n, int k, int p);

};

PetscErrorCode matrixKron::matMult(Mat dummy, Vec source, Vec target)
{
	PetscPrintf(PETSC_COMM_WORLD,"No sir, I DONT LIKE TO MOVE IT MOVE IT\n");

	PetscInt i,istart,iend;
	PetscInt irow, icol;
	PetscScalar v;
	PetscErrorCode ierr;

	matrixKron* local;
	MatShellGetContext(dummy,&local);

	//ierr = VecView(local->dia,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	// Reshape X matrix : Optimization could be done!!
	Mat XT;
	Mat tp1;
	Mat Y;
	ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,local->arows,local->bcols, NULL,&XT);CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(source,&istart,&iend);CHKERRQ(ierr);
  	for (i=istart; i<iend; i++) {
		irow = i/local->bcols;
		icol = i%local->bcols;
		ierr = VecGetValues(source,1,&i,&v);CHKERRQ(ierr);
		ierr = MatSetValue(XT,irow, icol, v,INSERT_VALUES);CHKERRQ(ierr);
  	}
	ierr = MatAssemblyBegin(XT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(XT,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	MatView(XT, PETSC_VIEWER_STDOUT_WORLD);

	ierr = MatMatTransposeMult(local->B,XT,MAT_INITIAL_MATRIX,PETSC_DEFAULT ,&tp1);
	ierr = MatMatTransposeMult(tp1,local->A,MAT_INITIAL_MATRIX,PETSC_DEFAULT ,&Y);

	MatView(tp1, PETSC_VIEWER_STDOUT_WORLD);
	MatView(Y, PETSC_VIEWER_STDOUT_WORLD);


	ierr = VecGetOwnershipRange(target,&istart,&iend);CHKERRQ(ierr);
  	for (i=istart; i<iend; i++) {
		irow = i/local->bcols;
		icol = i%local->bcols;
		ierr = VecGetValues(source,1,&i,&v);CHKERRQ(ierr);
		ierr = (target,i,v,INSERT_VALUES);
  	}
  	ierr = VecAssemblyBegin(target);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(target);CHKERRQ(ierr);


}

PetscErrorCode matrixKron::setupAB(int m, int n, int k, int l)
{
	arows = k;
	acols = l;
	brows = m;
	bcols = n;

	PetscScalar* v;
	PetscInt *J;
	PetscMalloc1(n, &v);
	PetscMalloc1(n, &J);

	PetscInt i,istart,iend;
	PetscScalar vs;
	PetscErrorCode ierr;
	int rank;
	

	// Create Matrix B
	ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,brows,bcols, NULL,&B);CHKERRQ(ierr);
	MatGetOwnershipRange(B,&istart,&iend);CHKERRQ(ierr);

	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	for (i=istart; i<iend; i++) { 
		for(int vx = 0; vx<bcols; vx++){
			v[vx] = i*bcols + vx +1;
			J[vx] = vx;
		}
		ierr = MatSetValues(B,1,&i,bcols,J,v,INSERT_VALUES);CHKERRQ(ierr);
   	}

	ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	MatView(B, PETSC_VIEWER_STDOUT_WORLD);

	
	// Create Matrix A
	ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,arows,acols, NULL,&A);CHKERRQ(ierr);
	MatGetOwnershipRange(A,&istart,&iend);CHKERRQ(ierr);
	vs = m*n;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	for (i=istart; i<iend; i++) { 
		for(int vx = 0; vx<acols; vx++){
			v[vx] = i*acols + vx +1 +vs;
			J[vx] = vx;
		}
		ierr = MatSetValues(A,1,&i,acols,J,v,INSERT_VALUES);CHKERRQ(ierr);
   	}

	ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	MatView(A, PETSC_VIEWER_STDOUT_WORLD);
}

PetscErrorCode MyKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscPrintf(PETSC_COMM_WORLD,"iteration %D KSP Residual norm %4.13e \n",n,rnorm);


  return 0;
}








int main(int argc,char **argv)
{
	
	PetscInt m,n,i,istart,iend;
	PetscScalar v;
	Vec testVector, resultVector;
	Mat A;
	matrixKron ctx;
	srand (time(NULL));
	int rank,size;
	PetscErrorCode ierr;

	PetscInitialize(&argc,&argv,(char*)0,"TEST");
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);
	srand (time(NULL));

	m = atoi(argv[1]);
	n = atoi(argv[2]);
	
	
	ctx.setupAB(2,2,2,2);
	PetscPrintf(PETSC_COMM_WORLD,"%d %d\n",m,n);
	

	// code that create test vector for matrix-product
	ierr = VecCreate(PETSC_COMM_WORLD,&testVector);CHKERRQ(ierr);
	ierr = VecSetSizes(testVector,PETSC_DECIDE,n);CHKERRQ(ierr);
	ierr = VecSetFromOptions(testVector);CHKERRQ(ierr);

	// Setting value for the testVector
	ierr = VecGetOwnershipRange(testVector,&istart,&iend);CHKERRQ(ierr);
  	for (i=istart; i<iend; i++) {
	    v    = (PetscReal)(i+1);
    	    ierr = VecSetValues(testVector,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  	}
  	ierr = VecAssemblyBegin(testVector);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(testVector);CHKERRQ(ierr);
	//ierr = VecView(testVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	
	ierr = VecDuplicate(testVector, &resultVector);CHKERRQ(ierr);



	PetscPrintf(PETSC_COMM_WORLD,"*********** Testing Setting for testVector************************\n",m,n);
	ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&ctx,&A);
	ierr = MatShellSetOperation(A, MATOP_MULT, (void (*)(void))matrixKron::matMult);
	ierr = VecView(testVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	MatMult(A,testVector,resultVector);
//	ierr = VecView(resultVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


/*
	PetscPrintf(PETSC_COMM_WORLD,"*********** Testing KSP solve ************************************\n",m,n);

	// Destroy the value in testVector
	ierr = VecGetOwnershipRange(testVector,&istart,&iend);CHKERRQ(ierr);
  	for (i=istart; i<iend; i++) {
	    v    = (PetscReal)((double)(rand()%10) + (double)(rand()%10)/100);
    	    ierr = VecSetValues(testVector,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  	}
  	ierr = VecAssemblyBegin(testVector);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(testVector);CHKERRQ(ierr);
	ierr = VecView(testVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	KSP ksp; 	
	//Create linear solver context
	ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
	// A as matrix and pre-conditioner
	ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
	// Set up tolerance and type of solver
	ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
	ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
	ierr = KSPGMRESCGSRefinementType(KSP_GMRES_CGS_REFINE_ALWAYS);
	// Set up what do to for monitoring convergence
	ierr = KSPMonitorSet(ksp,MyKSPMonitor,NULL,0);CHKERRQ(ierr);
	// Solve
	PetscPrintf(PETSC_COMM_WORLD,"sovling\n");
	ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
	ierr = KSPSolve(ksp,resultVector,testVector);CHKERRQ(ierr);
*/


	PetscPrintf(PETSC_COMM_WORLD,"FINAL\n");	
	ierr = PetscFinalize();
	return ierr;
}


	

#include "tensors.hpp"
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <stdlib.h> 


class matrixFormData
{
public:
	Vec dia;

// Using local data
};


PetscErrorCode MyKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *dummy)
{
  PetscPrintf(PETSC_COMM_WORLD,"iteration %D KSP Residual norm %4.13e \n",n,rnorm);
  return 0;
}

// first userMult simple set values
PetscErrorCode usermult(Mat dummy, Vec source, Vec target)
{
	PetscInt i,istart,iend;
	PetscScalar v;
	PetscErrorCode ierr;
	int rank;

	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	ierr = VecGetOwnershipRange(source,&istart,&iend);CHKERRQ(ierr);
  	for (i=istart; i<iend; i++) {
	    v    = (PetscReal)(rank);
    	    ierr = VecSetValues(target,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  	}
  	ierr = VecAssemblyBegin(target);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(target);CHKERRQ(ierr);
	return ierr;
}


// Useing the source values
PetscErrorCode usermult2(Mat dummy, Vec source, Vec target)
{
	PetscInt i,istart,iend;
	PetscScalar v;
	PetscErrorCode ierr;

	VecAXPBY(target,3,0,source);

  	ierr = VecAssemblyBegin(target);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(target);CHKERRQ(ierr);

	return ierr;
}

// Using user defined mult with private data
PetscErrorCode usermult3(Mat dummy, Vec source, Vec target)
{

	PetscInt i,istart,iend;
	PetscScalar v;
	PetscErrorCode ierr;

	matrixFormData* local;
	MatShellGetContext(dummy,&local);

	//ierr = VecView(local->dia,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	VecAXPBY(target,1,0, local->dia);
  	ierr = VecAssemblyBegin(target);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(target);CHKERRQ(ierr);
	ierr = VecView(target,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	return ierr;
}




int main(int argc,char **argv)
{
	
	PetscInt m,n,i,istart,iend;
	PetscScalar v;
	Vec testVector, resultVector;
	Mat A;
	matrixFormData ctx;
	srand (time(NULL));
	int rank,size;
	PetscErrorCode ierr;

	PetscInitialize(&argc,&argv,(char*)0,"TEST");
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);
	srand (time(NULL));

	m = atoi(argv[1]);
	n = atoi(argv[2]);

	

	PetscPrintf(PETSC_COMM_WORLD,"%d %d\n",m,n);
	

	// code that create serial/distribute vector to form matrix-product
	ierr = VecCreate(PETSC_COMM_WORLD,&(ctx.dia));CHKERRQ(ierr);
	ierr = VecSetSizes(ctx.dia,PETSC_DECIDE,n);CHKERRQ(ierr);
	ierr = VecSetFromOptions(ctx.dia);CHKERRQ(ierr);

	// Setting value for the diagonal.
	ierr = VecGetOwnershipRange(ctx.dia,&istart,&iend);CHKERRQ(ierr);
  	for (i=istart; i<iend; i++) {
	    v    = (PetscReal)(i+1);
    	    ierr = VecSetValues(ctx.dia,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  	}
	ierr = VecAssemblyBegin(ctx.dia);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(ctx.dia);CHKERRQ(ierr);
	//ierr = VecView(ctx.dia,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



	// code that create test vector for matrix-product
	ierr = VecCreate(PETSC_COMM_WORLD,&testVector);CHKERRQ(ierr);
	ierr = VecSetSizes(testVector,PETSC_DECIDE,n);CHKERRQ(ierr);
	ierr = VecSetFromOptions(testVector);CHKERRQ(ierr);

	// Setting value for the testVector
	ierr = VecGetOwnershipRange(testVector,&istart,&iend);CHKERRQ(ierr);
  	for (i=istart; i<iend; i++) {
	    v    = (PetscReal)(1);
    	    ierr = VecSetValues(testVector,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  	}
  	ierr = VecAssemblyBegin(testVector);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(testVector);CHKERRQ(ierr);
	//ierr = VecView(testVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	
	ierr = VecDuplicate(testVector, &resultVector);CHKERRQ(ierr);



	PetscPrintf(PETSC_COMM_WORLD,"*********** Testing Setting for testVector************************\n",m,n);
	ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&ctx,&A);
	ierr = MatShellSetOperation(A, MATOP_MULT, (void (*)(void))usermult);
	ierr = VecView(testVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	MatMult(A,testVector,resultVector);
	ierr = VecView(resultVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


	PetscPrintf(PETSC_COMM_WORLD,"*********** Testing scalar multiplication for testVector**********\n",m,n);
	ierr = MatShellSetOperation(A, MATOP_MULT, (void (*)(void))usermult2);
	ierr = VecView(testVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	MatMult(A,testVector,resultVector);
	ierr = VecView(resultVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	PetscPrintf(PETSC_COMM_WORLD,"*********** Testing diagonal multiplication for testVector********\n",m,n);
	ierr = MatShellSetOperation(A, MATOP_MULT, (void (*)(void))usermult3);
	ierr = VecView(testVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
	MatMult(A,testVector,resultVector);
	ierr = VecView(resultVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


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



	PetscPrintf(PETSC_COMM_WORLD,"FINAL\n");	
	ierr = PetscFinalize();
	return ierr;
}


	

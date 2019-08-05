//#include "tensors.hpp"
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <stdlib.h> 




class matrixKron
{
public:

	static PetscErrorCode matMult(Mat dummy, Vec source, Vec target);
	static PetscErrorCode apply_explicit();
	void setupAB();
	void preProcess();
	void postProcess();

	PDE<double> const pde;
	element_table const table;
	std::vector<fk::vector<double>> const &unscaled_sources;
	host_workspace<P> &host_space;
	rank_workspace<P> &rank_space;
	std::vector<element_chunk> chunks;
	double const time;
	double const dt;

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


	//Option 2: Modify host_space and rank space for each mulptilication
	// Pre process before multiplication
	preProcess();
	

	// Where the actual multiplication is done
	apply_explicit(pde, elem_table, chunks, host_space, rank_space);
	// I believe at the end of this function host_space contain correct result with correct ID?
	// Depending if you have the local data only, or overlaping data there is 2 way for updates

	postPorcess();



	// Accessing only local
	// setup start and end then do this localy
	ierr = VecGetOwnershipRange(target,&istart,&iend);CHKERRQ(ierr);
  	for (i=istart; i<iend; i++) {
				v = value of local vector(i)
				ierr = VecSetValues(target,i,v,INSERT_VALUES);
  	}
  ierr = VecAssemblyBegin(target);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(target);CHKERRQ(ierr);

	// Example for element wise access for all points
	// insert brainlessly  
	for (i=0; i<local_count; i++) {
		v = value of local vector(i)
		index = getIndex(i)
		ierr = VecSetValues(target,index,&i,&v,ADD_VALUES);CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(x);CHKERRQ(ierr);


}


PetscErrorCode matrixKron::setup()/preProcess()/postProcess
{
	//Option 1: Modify host_space and rank space 1 time for all multiplication
	fm::copy(host_space.x, host_space.x_orig);
  // see https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods
  P const a21 = 0.5;
  P const a31 = -1.0;
  P const a32 = 2.0;
  P const b1  = 1.0 / 6.0;
  P const b2  = 2.0 / 3.0;
  P const b3  = 1.0 / 6.0;
  P const c2  = 1.0 / 2.0;
  P const c3  = 1.0;

  scale_sources(pde, unscaled_sources, host_space.scaled_source, time);
  fm::axpy(host_space.scaled_source, host_space.fx);
  fm::copy(host_space.fx, host_space.result_1);
  P const fx_scale_1 = a21 * dt;
  fm::axpy(host_space.fx, host_space.x, fx_scale_1);
	
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
	srand (time(NULL));
	int rank,size;
	PetscErrorCode ierr;
	PetscInitialize(&argc,&argv,(char*)0,"TEST");
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	MPI_Comm_size(PETSC_COMM_WORLD,&size);
	srand (time(NULL));


	// Setup local data
	matrixKron ctx;	
	ctx.setupAB();


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
	//ierr = VecView(resultVector,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



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
	//ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
	
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


	

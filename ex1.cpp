
static char help[] = "Parallel vector layout.\n\n";

/*T
   Concepts: vectors^setting values
   Concepts: vectors^local access to
   Concepts: vectors^drawing vectors;
   Processors: n
T*/

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,istart,iend,n = 6,nlocal;
  PetscScalar    v,*array;
  Vec            x,y,z;
  PetscViewer    viewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /*
     Create a vector, specifying only its global dimension.
     When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
     the vector format (currently parallel or sequential) is
     determined at runtime.  Also, the parallel partitioning of
     the vector is determined by PETSc at runtime.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  
	// Example for element wise access for all points
  for (i=0; i<n; i++) {
    v    = (PetscReal)(rank);
    ierr = VecSetValues(x,1,&i,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);



  // Example for seting local
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(y,&istart,&iend);CHKERRQ(ierr);
  for (i=istart; i<iend; i++) {
    v    = (PetscReal)(rank+1);
    ierr = VecSetValues(y,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);




  /* --------------------------------------------------------------------
       Access the vector values directly. Each processor has access only
    to its portion of the vector. For default PETSc vectors VecGetArray()
    does NOT involve a copy
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&z);CHKERRQ(ierr);
  ierr = VecSetSizes(z,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(z);CHKERRQ(ierr);
  ierr = VecGetLocalSize(z,&nlocal);CHKERRQ(ierr);
  ierr = VecGetArray(z,&array);CHKERRQ(ierr);
  for (i=0; i<nlocal; i++) array[i] = rank;
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
  ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // vector operations:
  VecWAXPY(z, 3, x, y);
  ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       nsize: 2

TEST*/

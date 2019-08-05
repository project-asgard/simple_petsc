
static char help[] = "Tests MatMeshToDual()\n\n";

/*T
   Concepts: Mat^mesh partitioning
   Processors: n
T*/

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat             A, *localA;
  PetscErrorCode  ierr;
  PetscInt        Nvertices = 6;       /* total number of vertices */
  PetscInt        ncells    = 2;       /* number cells on this process */
  PetscInt        *ii,*jj;
  PetscMPIInt     size,rank;
  PetscInt       n,i,j,Ii,Istart,Iend;
  PetscScalar*    v;
  PetscInt *J;
  PetscInt t;
  //PetscReals values[3];
  MatPartitioning part;
  IS              is;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRQ(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  //ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,4,4, NULL,&A);CHKERRQ(ierr);
  MatSetUp(A);
  n=4;

  MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  PetscMalloc1(n, &v);
  PetscMalloc1(n, &J);

  for (Ii=Istart; Ii<Iend; Ii++) { 
    i = Ii;
    for(int vx = 0; vx<n; vx++){
      v[vx] = (PetscReal)(rank+1)*vx;
      J[vx] = vx;
    }
    ierr = MatSetValues(A,1,&i,4,J,v,INSERT_VALUES);CHKERRQ(ierr);
   
  }


  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscFinalize();
  return ierr;
}





/*TEST

   build:
     requires: parmetis

   test:
      nsize: 2
      requires: parmetis

TEST*/

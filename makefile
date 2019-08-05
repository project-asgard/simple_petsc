ALL: test
CC= mpicc
CXX = mpiCC
CFLAGS = -I${PETSC_DIR}/include -W
FFLAGS = -I${PETSC_DIR}/include/finclude
SOURCESC = ex7.cpp

OBJ = $(SOURCESC:.cpp=.o)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

test: ${OBJ} chkopts
	-${CLINKER} -o test ${OBJ} ${PETSC_SYS_LIB}

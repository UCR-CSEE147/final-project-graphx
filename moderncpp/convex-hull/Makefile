NVCC = nvc++
ifeq (,$(shell which nvprof))
NVCC_FLAGS = -O3 -arch=sm_70
else
NVCC_FLAGS = -O3 
endif
LD_FLAGS = -stdpar
EXE = moderncpp
OBJ = main.o

default: $(EXE)

main.o: main.cpp
	$(NVCC) -c -o $@ main.cpp $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
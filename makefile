FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody-cuda: nbody.o compute.cu
        nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody: nbody.o compute.o
        gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
        gcc $(FLAGS) -c $<
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
        gcc $(FLAGS) -c $<
clean:
      	rm -f *.o core.* nbody nbody-cuda
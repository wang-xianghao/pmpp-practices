PROGS := matmul 

LDFLAGS := -I ../include/

%: %.cu
	nvcc $(LDFLAGS) -o $@ $^ 

clean:
	rm -f *.o $(PROGS)
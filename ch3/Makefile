PROGS := matmul 

LDFLAGS := -I ../include/

matmul: matmul.cu
	nvcc $(LDFLAGS) -o $@ $^ 

clean:
	rm -f *.o $(PROGS)
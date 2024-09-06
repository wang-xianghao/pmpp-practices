PROGS := deviceQuery

LDFLAGS := -I ../include/
CFLAGS := -O2

all: $(PROGS)

%: %.cu
	nvcc $(CFLAGS) $(LDFLAGS) -o $@ $^ 

clean:
	rm -f *.o $(PROGS)
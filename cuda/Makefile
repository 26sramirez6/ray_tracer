CC=nvcc
CFLAGS=-arch=sm_37 -O3 -std=c++11

source = $(wildcard *.cu)
obj = $(source:.cu=.o)
exe = $(basename $(source))

all: clean $(exe)
	
$(obj): %.o : %.cu
	$(CC) $(CFLAGS) -c $< -o $@
	
$(exe): %: %.o
	$(CC) $(CFLAGS) -o $@ $<
	
clean:
	rm -f $(obj) $(exe)
CC=nvcc
CFLAGS=-arch=sm_37 -pedantic -Wall -Werror -Wextra -O3 -g -std=c++11 -fopenmp

source = $(wildcard *.cu)
obj = $(source:.cpp=.o)
exe = $(basename $(source))

all: clean $(exe)
	
$(obj): %.o : %.cu
	$(CC) $(CFLAGS) -c $< -o $@
	
$(exe): %: %.o
	$(CC) $(CFLAGS) -o $@ $<
	
clean:
	rm -f $(obj) $(exe)
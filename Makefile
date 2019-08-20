CC=g++
CFLAGS=-pedantic -Wall -Werror -Wextra -O3 -g -std=c++11 -Winline -fopenmp #-DNSERIALIZE

source = $(wildcard *.cpp)
obj = $(source:.cpp=.o)
exe = $(basename $(source))

all: clean $(exe)
	
$(obj): %.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
	
$(exe): %: %.o
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(obj) $(exe)
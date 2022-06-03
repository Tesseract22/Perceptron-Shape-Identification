CXX=clang++
CXXFLAGS=-std=c++20 -g -fstandalone-debug -Wall -Wextra -Werror -pedantic

exec: bin/exec
clean:
	rm -fr bin/* debug/*

bin/exec: main.cc
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: clean exec

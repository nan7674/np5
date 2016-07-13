.PHONY: clean

SRC = ../src/polynomial.cpp ../utils/data.cpp
HDR = ../src/common.hpp ../src/polynomial.hpp ../utils/data.hpp
CCOPT = -Ofast -march=native

examples: examples.hpp examples.cpp main.cpp ${HDR} ${SRC}
	g++ -std=c++11 ${CCOPT} -I../src -I../ examples.cpp main.cpp \
	    ../utils/data.cpp ../src/mcore/calc.cpp ../src/mcore/linalg.cpp \
	    ../src/polynomial.cpp -o examples

clean:
	rm -f examples

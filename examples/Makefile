.PHONY: clean

SRC = \
    ../src/polynomial.cpp \
    ../src/spline.cpp \
    ../src/mcore/calc.cpp \
    ../src/mcore/linalg.cpp \
    ../utils/data.cpp

HDR = \
    ../src/common.hpp \
    ../src/polynomial.hpp \
    ../src/spline.hpp \
    ../src/mcore/calc.hpp \
    ../src/mcore/linalg.hpp \
    ../utils/data.hpp

CCOPT = -O2 -march=native -flto -fno-rtti
#-Ofast -march=native

examples: examples.hpp examples.cpp main.cpp ${HDR} ${SRC}
	g++ -std=c++11 ${CCOPT} -I../src -I../ examples.cpp main.cpp \
	    ${SRC} -llapack -o examples

clean:
	rm -f examples

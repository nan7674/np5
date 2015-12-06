.PHONY: clean

DEPS = \
    ../src/mcore/sequence.hpp \
    ../src/mcore/linalg.hpp \
    ../src/mcore/calc.hpp \
    ../src/mcore/optimization.hpp \
    ../src/polynomial.hpp

SRC = ../src/mcore/calc.cpp ../src/mcore/linalg.cpp
INC = -I../src/
LIB = -L/usr/lib64
CC = g++
CC_OPTS = -std=c++11 -Ofast -march=native
#CC = clang++
#CC_OPTS = -std=c++11 --analyze
#CC_OPTS = -std=c++11 -fsanitize=memory -fsanitize-memory-track-origins=2 -g
ut: $(DEPS) main.cpp $(SRC)
	$(CC) $(CC_OPTS) $(INC) main.cpp $(SRC) -lboost_unit_test_framework -o ut

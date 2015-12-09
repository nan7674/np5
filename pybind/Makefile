
INC = -I/usr/include/python2.7 -I../src/
DEPS = ../src/mcore/linalg.hpp ../src/polynomial.hpp
LIB_FILE = regression.cpp
CPP_FLAGS = -std=c++11 -Ofast -march=native -DNDEBUG -Wall

regression.so: regression.cpp $(DEPS)
	g++ $(CPP_FLAGS) -shared -fPIC $(INC) regression.cpp ../src/mcore/linalg.cpp -o -lpython -o regression.so

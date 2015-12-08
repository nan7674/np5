
INC = -I/usr/include/python2.7 -I../src/
LIB_FILE = regression.cpp

regression.so: regression.cpp
	g++ -std=c++11 -shared -fPIC $(INC) regression.cpp ../src/mcore/linalg.cpp -o -lpython -o regression.so

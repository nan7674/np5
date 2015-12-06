.PHONY: clean

CCFLAGS = -std=c++11 -O3 -march=native -DNDEBUG

build/:
	mkdir build

clean:
	rm -rf build/
	rm -f demo libreg.a src/*~ examples/*~ *~

libreg.a: build/ build/spline.o build/calc.o build/linalg.o
	ar rvc libreg.a build/spline.o build/calc.o build/linalg.o

build/spline.o: build/ src/spline.hpp src/spline.cpp
	g++ -c $(CCFLAGS) src/spline.cpp -Isrc/ -o build/spline.o

build/calc.o: build/ src/mcore/calc.hpp src/mcore/calc.cpp
	g++ -c $(CCFLAGS) src/mcore/calc.cpp -o build/calc.o

build/linalg.o: build src/mcore/linalg.hpp src/mcore/linalg.cpp
	g++ -c $(CCFLAGS) src/mcore/linalg.cpp -o build/linalg.o


demo: libreg.a examples/examples.hpp examples/examples.cpp examples/main.cpp
	g++ $(CCFLAGS) -L. -I. examples/examples.cpp examples/main.cpp -lreg  -o demo




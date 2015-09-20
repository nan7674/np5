.PHONY: clean

CCFLAGS = -std=c++11 -ggdb

build/:
	mkdir build
	
clean:
	rm -rf build/
	rm -f demo libreg.a src/*~ examples/*~ *~

libreg.a: build/ build/spline.o build/mcore.o
	ar rvc libreg.a build/spline.o build/mcore.o

build/spline.o: build/ src/spline.hpp src/spline.cpp
	g++ -c $(CCFLAGS) src/spline.cpp -Isrc/ -o build/spline.o

build/mcore.o: build/ src/mcore.hpp src/mcore.cpp
	g++ -c $(CCFLAGS) src/mcore.cpp -o build/mcore.o

demo: libreg.a examples/examples.hpp examples/examples.cpp examples/main.cpp
	g++ $(CCFLAGS)  -L. -I. examples/examples.cpp examples/main.cpp -lreg  -o demo 




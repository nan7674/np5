.PHONY: clean optimized

DEPS = \
    ../src/mcore/sequence.hpp \
    ../src/mcore/linalg.hpp \
    ../src/mcore/calc.hpp \
    ../src/mcore/optimization.hpp \
    ../src/polynomial.hpp \
    ../src/spline.hpp \
    ../utils/data.hpp

SRC = \
    ../src/mcore/calc.cpp \
    ../src/mcore/linalg.cpp \
    ../src/polynomial.cpp \
    ../src/spline.cpp \
    ../utils/data.cpp

INC = -I../src/ -I../
LIB = -L/usr/lib64
CC_GCC = g++
CC_GCC_OPTS = -std=c++11

# -Ofast -march=native -DNDEBUG
CC_CLANG = clang++
#CC_CLANG_OPTS = -std=c++11 --analyze
CC_CLANG_OPTS = -std=c++11 -fsanitize=memory -fsanitize-memory-track-origins=2 -g

comp = gcc
ifeq ($(comp), gcc)
	CC_COMP = $(CC_GCC)
	OPTS = $(CC_GCC_OPTS)
else
	CC_COMP = $(CC_CLANG)
	OPTS = $(CC_CLANG_OPTS)
endif

#optimize = false
#ifeq ($(optimize), true)

ut: $(DEPS) main.cpp $(SRC)
	@echo "compiler : " $(comp)
	@echo "options :" $(OPTS)
	$(CC_COMP) $(OPTS) $(INC) main.cpp $(SRC) -llapack -lboost_unit_test_framework -o ut

clean:
	rm -f ut ut.log

optimized:
	@echo "Create an optimized version"
	rm ut
	$(eval OPTS := $(OPTS) -Ofast)
	$(MAKE) ut
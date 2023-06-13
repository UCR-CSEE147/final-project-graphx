CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3

SRC = main.cpp	
OBJ = $(SRC:.cpp=.o)
EXE = naive

.PHONY: all clean

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(EXE)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXE)

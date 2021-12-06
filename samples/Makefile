CCOMP=g++

STD_CCOMP_FLAGS=-std=c++17 -Wall -pedantic -g
CCOMP_FLAGS=$(STD_CCOMP_FLAGS)
CLINK_FLAGS=-Wall

STD_LIBS=-lrt -lm
GRAPHICS_LIBS=-lsfml-graphics -lsfml-window -lsfml-system
SPIKES_COUNT_LIBS=-lncurses
TIME_CUDA_LIBS=-lcuda -lcudart -lcublas -lcurand
LIATH_LIBS=-lliath

SRC_DIR=./src
BLD_DIR=./bld
BIN_DIR=./bin

# Adds BLD_DIR to object parameter names.
OBJS=$(patsubst %.o,$(BLD_DIR)/%.o,$@.o)

MKDIR=mkdir -p
RM=rm -rf

all: clean create terminal graphics

terminal: create
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(LIATH_LIBS)
	@printf "\nCreated terminal!\n\n"

graphics: create
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(GRAPHICS_LIBS) $(LIATH_LIBS)
	@printf "\nCreated graphics!\n\n"

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

clean:
	$(RM) $(BLD_DIR)
	$(RM) $(BIN_DIR)
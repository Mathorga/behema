CCOMP=g++
NVCOMP=nvcc

STD_CCOMP_FLAGS=-std=c++17 -Wall -pedantic -g
CCOMP_FLAGS=$(STD_CCOMP_FLAGS)
CLINK_FLAGS=-Wall

ifdef CUDA_ARCH
CUDA_ARCH_FLAG=-arch=$(CUDA_ARCH)
else
CUDA_ARCH_FLAG=
endif

NVCOMP_FLAGS=-std=c++11 -G $(CUDA_ARCH_FLAG)
NVLINK_FLAGS=$(CUDA_ARCH_FLAG)

STD_LIBS=-lrt -lm
CUDA_STD_LIBS=-lcudart -lstdc++
behema_LIBS=-lbehema

SRC_DIR=./src
BLD_DIR=./bld
BIN_DIR=./bin
OUT_DIR=./out

# Adds BLD_DIR to object parameter names.
OBJS=$(patsubst %.o,$(BLD_DIR)/%.o,$@.o)

MKDIR=mkdir -p
RM=rm -rf

all: bench

bench: create
	@printf "\n"
	$(NVCOMP) $(NVCOMP_FLAGS) -c $(SRC_DIR)/$@.cu -o $(BLD_DIR)/$@.o
	$(NVCOMP) $(NVLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(CUDA_STD_LIBS) $(behema_LIBS)
	@printf "\nCreated $@!\n"

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)
	$(MKDIR) $(OUT_DIR)

clean:
	$(RM) $(BLD_DIR)
	$(RM) $(BIN_DIR)
	$(RM) $(OUT_DIR)

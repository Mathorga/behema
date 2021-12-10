CCOMP=gcc
NVCOMP=nvcc

STD_CCOMP_FLAGS=-std=c17 -Wall -pedantic -g
CCOMP_FLAGS=$(STD_CCOMP_FLAGS)
CLINK_FLAGS=-Wall

ifdef CUDA_ARCH
CUDA_ARCH_FLAG=-arch=$(CUDA_ARCH)
else
CUDA_ARCH_FLAG=
endif

NVCOMP_FLAGS=--compiler-options '-fPIC' -G $(CUDA_ARCH_FLAG)
NVLINK_FLAGS=$(CUDA_ARCH_FLAG)

STD_LIBS=-lrt -lm
CUDA_STD_LIBS=-lcudart
LIBS=$(STD_LIBS)

SRC_DIR=./src
BLD_DIR=./bld
BIN_DIR=./bin

SYSTEM_INCLUDE_DIR=/usr/include
SYSTEM_LIB_DIR=/usr/lib

# Adds BLD_DIR to object parameter names.
OBJS=$(patsubst %.o,$(BLD_DIR)/%.o,$^)

MKDIR=mkdir -p
RM=rm -rf

# Installs the library files (headers and compiled) into the default system lookup folders.
all: std-install

std-install: create std
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/liath
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/liath
	sudo cp $(BLD_DIR)/libliath.so $(SYSTEM_LIB_DIR)
	@printf "\nInstallation complete!\n"

cuda-install: create cuda
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/liath
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/liath
	sudo cp $(BLD_DIR)/libliath.so $(SYSTEM_LIB_DIR)
	@printf "\nInstallation complete!\n"

uninstall: clean
	sudo $(RM) $(SYSTEM_INCLUDE_DIR)/liath
	sudo $(RM) $(SYSTEM_LIB_DIR)/libliath.so
	sudo $(RM) $(SYSTEM_LIB_DIR)/libliath.a
	@printf "\nSuccessfully uninstalled.\n"


# Builds all library files.
std: liath_std.o
	$(CCOMP) $(CLINK_FLAGS) -shared $(OBJS) -o $(BLD_DIR)/libliath.so

cuda: liath_cuda.o
	$(NVCOMP) $(NVLINK_FLAGS) -shared $(OBJS) $(CUDA_STD_LIBS) -o $(BLD_DIR)/libliath.so



# Builds object files from source.
%.o: $(SRC_DIR)/%.c
	$(CCOMP) $(CCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@

%.o: $(SRC_DIR)/%.cu
	$(NVCOMP) $(NVCOMP_FLAGS) -c $^ -o $(BLD_DIR)/$@



# Creates temporary working directories.
create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)

# Removes temporary working directories.
clean:
	$(RM) $(BLD_DIR)
	$(RM) $(BIN_DIR)
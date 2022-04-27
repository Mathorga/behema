CCOMP=gcc
NVCOMP=nvcc

STD_CCOMP_FLAGS=-std=c17 -Wall -pedantic -g
CCOMP_FLAGS=$(STD_CCOMP_FLAGS) -fopenmp
CLINK_FLAGS=-Wall -fopenmp

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

all: std

install: std-install

# Installs the library files (headers and compiled) into the default system lookup folders.
std-install: std
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/portia
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/portia
	sudo cp $(BLD_DIR)/libportia.so $(SYSTEM_LIB_DIR)
	@printf "\nInstallation complete!\n\n"

cuda-install: cuda
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/portia
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/portia
	sudo cp $(BLD_DIR)/libportia.so $(SYSTEM_LIB_DIR)
	@printf "\nInstallation complete!\n\n"

uninstall: clean
	sudo $(RM) $(SYSTEM_INCLUDE_DIR)/portia
	sudo $(RM) $(SYSTEM_LIB_DIR)/libportia.so
	sudo $(RM) $(SYSTEM_LIB_DIR)/libportia.a
	@printf "\nSuccessfully uninstalled.\n\n"

std: create std-build
cuda: create cuda-build


# Builds all library files.
std-build: cortex.o portia_std.o utils.o
	$(CCOMP) $(CLINK_FLAGS) -shared $(OBJS) -o $(BLD_DIR)/libportia.so
	@printf "\nCompiled $@!\n\n"

cuda-build: cortex.o portia_cuda.o utils.o
	$(NVCOMP) $(NVLINK_FLAGS) -shared $(OBJS) $(CUDA_STD_LIBS) -o $(BLD_DIR)/libportia.so
	@printf "\nCompiled $@!\n\n"



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

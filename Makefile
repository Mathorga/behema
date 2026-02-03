# Default C compiler.
CCOMP=gcc

# Default CUDA compiler.
NVCOMP=nvcc

# Default archive utility.
ARC=ar

##################### OS Detection & Configuration #####################

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    include darwin.mk
else
    # Default to Linux if not Mac
    include linux.mk
endif

##################### Global Settings #####################

STD_CCOMP_FLAGS=-std=c11 -Wall -pedantic -g -fPIC
CCOMP_FLAGS=$(STD_CCOMP_FLAGS) -fopenmp
CLINK_FLAGS=-Wall -fopenmp
ARC_FLAGS=-rcs

ifdef CUDA_ARCH
    CUDA_ARCH_FLAG=-arch=$(CUDA_ARCH)
else
    CUDA_ARCH_FLAG=
endif

# Mode flag: if set to "archive", installs behema as a static library.
MODE=

NVCOMP_FLAGS=--compiler-options '-fPIC' -G $(CUDA_ARCH_FLAG)
NVLINK_FLAGS=$(CUDA_ARCH_FLAG)

STD_LIBS=-lm $(STD_LIBS_EXTRA)
CUDA_STD_LIBS=-lcudart

SRC_DIR=./src
BLD_DIR=./bld
BIN_DIR=./bin

# Adds BLD_DIR to object parameter names.
OBJS=$(patsubst %.o,$(BLD_DIR)/%.o,$^)

MKDIR=mkdir -p
RM=rm -rf

##################### Targets #####################

all: std

install: std-install

# Installs all header files to the default include dir.
install-headers:
	@printf "\nInstalling headers...\n\n"
	sudo $(MKDIR) $(SYSTEM_INCLUDE_DIR)/behema
	sudo cp $(SRC_DIR)/*.h $(SYSTEM_INCLUDE_DIR)/behema


# Installs the generated lib file to the default lib dir.
install-lib:
ifneq ($(MODE), archive)
	@printf "\nInstalling dynamic library ($(LIB_EXT))...\n\n"
	sudo cp $(BLD_DIR)/libbehema$(LIB_EXT) $(SYSTEM_LIB_DIR)
endif
ifeq ($(MODE), archive)
	@printf "\nInstalling static library...\n\n"
	sudo cp $(BLD_DIR)/libbehema.a $(SYSTEM_LIB_DIR)
endif


# Installs the library files (headers and compiled) into the default system lookup folders.
std-install: std install-headers install-lib
	@printf "\nInstallation complete!\n\n"

cuda-install: cuda install-headers install-lib
	@printf "\nInstallation complete!\n\n"


# Uninstalls any previous installation.
uninstall: clean
	sudo $(RM) $(SYSTEM_INCLUDE_DIR)/behema
	sudo $(RM) $(SYSTEM_LIB_DIR)/libbehema$(LIB_EXT)
	sudo $(RM) $(SYSTEM_LIB_DIR)/libbehema.a
	@printf "\nSuccessfully uninstalled.\n\n"

std: create std-build
cuda: create cuda-build

# Builds all library files.
# NOTE: We use $(LDFLAGS_LIB) instead of hardcoded -shared
std-build: cortex.o utils.o population.o behema_std.o
	$(CCOMP) $(CLINK_FLAGS) $(LDFLAGS_LIB) $(OBJS) $(STD_LIBS) -o $(BLD_DIR)/libbehema$(LIB_EXT)
	$(ARC) $(ARC_FLAGS) $(BLD_DIR)/libbehema.a $(OBJS)
	@printf "\nCompiled $@!\n"

cuda-build: cortex.o utils.o population.o behema_cuda.o
	$(NVCOMP) $(NVLINK_FLAGS) $(NV_LDFLAGS_LIB) $(OBJS) $(CUDA_STD_LIBS) -o $(BLD_DIR)/libbehema$(LIB_EXT)
	$(ARC) $(ARC_FLAGS) $(BLD_DIR)/libbehema.a $(OBJS)
	@printf "\nCompiled $@!\n"


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
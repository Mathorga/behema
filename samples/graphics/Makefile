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
SFML_LIBS=-lsfml-graphics -lsfml-window -lsfml-system
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio
SPIKES_COUNT_LIBS=-lncurses
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

all: clean camera inspect

camera: create
	@printf "\n"
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o -I/usr/include/opencv4
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(OPENCV_LIBS) $(SFML_LIBS) $(behema_LIBS)
	@printf "\nCreated $@!\n"

cuda_camera: create
	@printf "\n"
	$(NVCOMP) $(NVCOMP_FLAGS) -c $(SRC_DIR)/$@.cu -o $(BLD_DIR)/$@.o -I/usr/include/opencv4
	$(NVCOMP) $(NVLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(CUDA_STD_LIBS) $(OPENCV_LIBS) $(SFML_LIBS) $(behema_LIBS)
	@printf "\nCreated $@!\n"

inspect: create
	@printf "\n"
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o -I/usr/include/opencv4
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(SFML_LIBS) $(behema_LIBS)
	@printf "\nCreated $@!\n"

create:
	$(MKDIR) $(BLD_DIR)
	$(MKDIR) $(BIN_DIR)
	$(MKDIR) $(OUT_DIR)

clean:
	$(RM) $(BLD_DIR)
	$(RM) $(BIN_DIR)
	$(RM) $(OUT_DIR)

reset:
	$(RM) $(OUT_DIR)

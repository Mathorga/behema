CCOMP=gcc

STD_CCOMP_FLAGS=-std=c11 -Wall -pedantic -g
CCOMP_FLAGS=$(STD_CCOMP_FLAGS) -I/usr/local/include
CLINK_FLAGS=-Wall

STD_LIBS=-lm
SFML_LIBS=-lsfml-graphics -lsfml-window -lsfml-system
RAYLIB_LIBS=-lraylib -lGL -lX11
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio
SPIKES_COUNT_LIBS=-lncurses
CUDA_STD_LIBS=-lcudart -lstdc++
BEHEMA_LIBS=-lbehema

SRC_DIR=./src
BLD_DIR=./bld
BIN_DIR=./bin
OUT_DIR=./out

# Adds BLD_DIR to object parameter names.
OBJS=$(patsubst %.o,$(BLD_DIR)/%.o,$@.o)

MKDIR=mkdir -p
RM=rm -rf

# Check what the current operating system is.
UNAME_S=$(shell uname -s)

# The curren OS is Linux.
ifeq ($(UNAME_S),Linux)
    STD_LIBS+=-lrt

    # Statically add raylib.
    RAYLIB_DIR=../libs/raylib/raylib-6.0_linux_amd64
    CCOMP_FLAGS+=-I$(RAYLIB_DIR)/include
    CLINK_FLAGS+=$(RAYLIB_DIR)/lib/libraylib.a
endif

# The current OS is MacOS.
ifeq ($(UNAME_S),Darwin)
    CCOMP_FLAGS+=-I/usr/local/include

    # Make sure libraries are found during linking (-L) and embedded in the binary (-rpath).
    CLINK_FLAGS+=-L/usr/local/lib -Wl,-rpath,/usr/local/lib

    # Statically add raylib.
    RAYLIB_DIR=../../libs/raylib/raylib-6.0_macos
    CCOMP_FLAGS+=-I$(RAYLIB_DIR)/include
    CLINK_FLAGS+=$(RAYLIB_DIR)/lib/libraylib.a
    CLINK_FLAGS+=-framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL

    # Statically add ccap.
    CCAP_DIR=../../libs/ccap/ccap-macos-universal-static
    CCOMP_FLAGS+=-I$(CCAP_DIR)/include
    CLINK_FLAGS+=$(CCAP_DIR)/lib/libccap.a
    CLINK_FLAGS+=-framework AVFoundation -framework CoreMedia -framework CoreVideo -framework Accelerate -framework Foundation -lc++
endif

all: clean camera inspect

camera: create
	@printf "\n"
	$(CPPCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o -I/usr/include/opencv4
	$(CPPCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(OPENCV_LIBS) $(SFML_LIBS) $(BEHEMA_LIBS)
	@printf "\nCreated $@!\n"

c_camera: create
	@printf "\n"
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.c -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(BEHEMA_LIBS)
	@printf "\nCreated $@!\n"

inspect: create
	@printf "\n"
	$(CPPCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.cpp -o $(BLD_DIR)/$@.o -I/usr/include/opencv4
	$(CPPCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(SFML_LIBS) $(BEHEMA_LIBS)
	@printf "\nCreated $@!\n"

raylib_inspect: create
	@printf "\n"
	$(CCOMP) $(CCOMP_FLAGS) -c $(SRC_DIR)/$@.c -o $(BLD_DIR)/$@.o
	$(CCOMP) $(CLINK_FLAGS) $(OBJS) -o $(BIN_DIR)/$@ $(STD_LIBS) $(RAYLIB_LIBS) $(BEHEMA_LIBS)
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

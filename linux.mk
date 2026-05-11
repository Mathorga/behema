# --- Linux Configuration ---
LIB_EXT=.so
HDR_DST_DIR=/usr/include
LIB_DST_DIR=/usr/lib
STD_LIBS_EXTRA=-lrt
INSTALL_NAME_FLAGS=

# Linker flags for C (GCC)
# We use -shared and set the SONAME
LDFLAGS_LIB=-shared -Wl,-soname,libbehema$(LIB_EXT)

# Linker flags for CUDA (NVCC)
# NVCC passes these to the linker
NV_LDFLAGS_LIB=-shared -Xlinker -soname,libbehema$(LIB_EXT)
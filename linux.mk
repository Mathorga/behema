# --- Linux Configuration ---
LIB_EXT=.so
SYSTEM_INCLUDE_DIR=/usr/include
SYSTEM_LIB_DIR=/usr/lib
STD_LIBS_EXTRA=-lrt

# Linker flags for C (GCC)
# We use -shared and set the SONAME
LDFLAGS_LIB=-shared -Wl,-soname,libbehema$(LIB_EXT)

# Linker flags for CUDA (NVCC)
# NVCC passes these to the linker
NV_LDFLAGS_LIB=-shared -Xlinker -soname,libbehema$(LIB_EXT)
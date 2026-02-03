# --- MacOS Configuration ---
LIB_EXT=.dylib
SYSTEM_INCLUDE_DIR=/usr/local/include
SYSTEM_LIB_DIR=/usr/local/lib
STD_LIBS_EXTRA=

# Linker flags for C (GCC)
# We use -dynamiclib and set the Install Name to @rpath
LDFLAGS_LIB=-dynamiclib -install_name @rpath/libbehema$(LIB_EXT)

# Linker flags for CUDA (NVCC)
# On MacOS, NVCC needs to pass -dynamiclib and install_name via -Xlinker
NV_LDFLAGS_LIB=-Xlinker -dynamiclib -Xlinker -install_name -Xlinker @rpath/libbehema$(LIB_EXT)
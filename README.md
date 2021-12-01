# Liath
<p align="center" width="100%">
    <img width="33%" src="/liath.png"> 
</p>
Spiking neural network implementation through cellular automata.

## Shared library installation (Linux)
Run `make` or `make all` to install all packages in a system-wide dynamic library.<br/>

### Standard
Run `make standard` to install the CPU package as a system-wide dynamic library.<br/>

### Uninstall
Run `make uninstall` to uninstall any previous installation.

WARNING: Every time you `make` a new package the previous installation is overwritten.

## How to use
### Header files
Once the installation is complete you can include the library by `#include <liath/liath.h>` and directly use every function in the packages you compiled.<br/>

### Linking
During linking you can specify `-lliath` in order to link the compiled functions.

## TODO
CUDA support

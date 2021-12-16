# Liath
<p align="center" width="100%">
    <img width="33%" src="/liath.png"> 
</p>
Spiking neural network implementation through cellular automata.

## Shared library installation (Linux)
### Standard
Run `make install` or `make std-install` to install the default (CPU) package in a system-wide dynamic library.<br/>

### CUDA
TODO

### Uninstall
Run `make uninstall` to uninstall any previous installation.

WARNING: Every time you `make` a new package the previous installation is overwritten.

## How to use
### Header files
Once the installation is complete you can include the library by `#include <liath/liath.h>` and directly use every function in the packages you compiled.<br/>

### Linking
During linking you can specify `-lliath` in order to link the compiled functions.

## TODO
Neurons competition for synapses
CUDA support

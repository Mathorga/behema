<p align="center" width="100%">
    <img width="66%" src="/meta/portia.png"> 
</p>
BEHEMA (BEHavioral EMergent Automaton) is a spiking neural network library inspired by cellular automata.<br/>
Behema borrows concepts such as grid layout and kernels from cellular automata to boost efficiency in highly parallel environments.
The implementations aims at mimicking a biological brain as closely as possible without losing performance. The learning pattern of a Behema neural network is continuos, with no distinction between training, validation and deploy. The network is continuously changed by its inputs and therefore can produce unexpected (emerging) results.

## Shared library installation (Linux)
### Standard
Run `make install` or `make std-install` to install the default (CPU) package in a system-wide dynamic library.<br/>

### CUDA
Run `make cuda-install` to install the CUDA parallel (GPU) package in a system-wide dynamic library.<br/>
Warnings:<br/>
* The CUDA version only works with NVIDIA GPUS<br/>
* The CUDA version requires the CUDA SDK and APIs to work<br/>
* The CUDA SDK or APIs are not included in any install_deps.sh script<br/>

### OpenCL
TODO

### Uninstall
Run `make uninstall` to uninstall any previous installation.

WARNING: Every time you `make` a new package the previous installation is overwritten.

## How to use
### Header files
Once the installation is complete you can include the library by `#include <portia/portia.h>` and directly use every function in the packages you compiled.<br/>

### Linking
During linking you can specify `-lportia` in order to link the compiled functions.

### Usage example
The first step is to create and initialize two cortexes by:
```
// Define starting parameters.
cortex_size_t cortex_width = 100;
cortex_size_t cortex_height = 60;
nh_radius_t nh_radius = 2;

// Define the sampling interval, used later.
ticks_count_t sampleWindow = 10;

// Create the cortexes.
cortex2d_t even_cortex;
cortex2d_t odd_cortex;

// Initialize the first cortex and copy it to the second.
c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
odd_cortex = *c2d_copy(&even_cortex);
```
This will setup two identical 100x60 cortexes with default values.<br/>
Optionally, before copying the first cortex to the second, its properties can be set by:
```
c2d_set_evol_step(&even_cortex, 0x20U);
c2d_set_pulse_window(&even_cortex, 0x3A);
c2d_set_syngen_beat(&even_cortex, 0.1F);
c2d_set_max_touch(&even_cortex, 0.2F);
c2d_set_sample_window(&even_cortex, sampleWindow);
```
The two cortexes will be updated alternatively at each iteration step.

Now the cortex can already be deployed, but it's often useful to setup its inputs and outputs:
```
// Support variable for input sampling.
ticks_count_t samplingBound = sampleWindow - 1;

// Define an input rectangle (the area of neurons directly attached to inputs).
// Since even_cortex and odd_cortex are 2D cortexes, inputs are arranged in a 2D surface.
// inputCoords contains the bound coordinates of the input rectangle as [x0, y0, x1, y1].
cortex_size_t inputsCoords[] = {10, 5, 40, 20};

// Allocate inputs according to the defined area.
ticks_count_t* inputs = (ticks_count_t*) malloc((inputsCoords[2] - inputsCoords[0]) * (inputsCoords[3] - inputsCoords[1]) * sizeof(ticks_count_t));

// Support variable used to keep track of the current step in the sampling window.
ticks_count_t sample_step = samplingBound;
```

#### Input mapping
<img width="33%" src="/meta/10f.png"> <img width="33%" src="/meta/10r.png">

## TODO
Neurons competition for synapses

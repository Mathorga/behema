/*
*****************************************************************
hal.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __HAL__
#define __HAL__

#include "field.h"

#ifdef __CUDACC__
#include "hal_cuda.h"
#else
#include "hal_std.h"
#endif

#endif

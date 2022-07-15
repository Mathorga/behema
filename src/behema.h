/*
*****************************************************************
behema.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __behema__
#define __behema__

#include "cortex.h"
#include "utils.h"

#ifdef __CUDACC__
#include "behema_cuda.h"
#else
#include "behema_std.h"
#endif

#endif

/*
*****************************************************************
kuiper.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __KUIPER__
#define __KUIPER__

#include "utils.h"
#include "neuron.h"

#ifdef __CUDACC__
#include "kuiper_cuda.h"
#else
#include "kuiper_std.h"
#endif

#endif
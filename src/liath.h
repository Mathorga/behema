/*
*****************************************************************
liath.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __KUIPER__
#define __KUIPER__

#include "field.h"

#ifdef __CUDACC__
#include "liath_cuda.h"
#else
#include "liath_std.h"
#endif

#endif
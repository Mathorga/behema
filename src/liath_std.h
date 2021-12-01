/*
*****************************************************************
liath_std.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __LIATH_STD__
#define __LIATH_STD__

#include <stdint.h>
#include <stdlib.h>
// TODO Remove in release.
#include <stdio.h>
#include <time.h>
#include "neuron.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialization functions:

/// Initializes the given field with default values.
void field_init(neuron_t* field);


// Execution functions:

/// Feeds external spikes to the specified neurons.
void field_feed(neuron_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value);

/// Propagates synapse spikes according to their progress.
void field_propagate(neuron_t* field);

/// Increments neuron values with spikes from input synapses.
void field_increment(neuron_t* field);

/// Decrements all neurons values by decay.
void field_decay(neuron_t* field);

/// Triggers neuron firing if values exceeds threshold.
void field_fire(neuron_t* field);

/// Relaxes value to neurons that exceeded their threshold.
void field_relax(neuron_t* field);

/// Performs a full run cycle over the network braph.
void field_tick(neuron_t* field);


// Learning functions:

/// Deletes all unused synapses.
void field_syndel(neuron_t* field);

/// Adds synapses to busy neurons (those that fire frequently).
void field_syngen(neuron_t* field);

/// Performs a full evolution cycle over the network braph.
void field_evolve(neuron_t* field);


#ifdef __cplusplus
}
#endif

#endif
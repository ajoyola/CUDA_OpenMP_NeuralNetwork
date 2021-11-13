/*
* Struct to define a single layer in a NN
*/
#ifndef LAYER_CUH
#define LAYER_CUH


#include <stdlib.h>
#include <stdio.h>

typedef struct Layer {
    double bias; // bias of the current layer
    double** weights; // 2D array of random weights
    int n_neurons; // number of output neurons
} Layer; 


/* Definition of the functions
*  More details of each function can be found in the layer.c
*/
Layer new_layer(int r, int size_out); //creates a new layer
double** generate_rnd_weights(int r, int size_out); //creates random weights for the layer
void free_layer(Layer layer); // deallocate a layer's memory

#endif
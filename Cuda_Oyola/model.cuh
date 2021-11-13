/*
* Struct to define a neural network
*/
#ifndef MODEL_CUH
#define MODEL_CUH


#include <stdio.h>
#include <stdlib.h>
#include "layer.cuh"

typedef struct Model {
    int r; //the constant R received in the main 
    int size_nn; // total number of layers
    int size_input; //size of the input given to the model
    Layer* layers; //1D array of typedef layer
} Model;

/* Definition of the functions
*  More details of each function can be found in the model.c
*/
Model new_model(int r, int size_nn, int size_input); //creates a new model
Layer* layers_init(int r, int size_nn, int size_input); //initializing the 1D array of layers of the model
double* predict(Model m, double* x, char type[], double* t_tp); // predict function for computing the output
void free_model(Model nn); // deallocate a model's memory
double get_throughput(int output_size, int r, float kernel_elapsedtime);
#endif
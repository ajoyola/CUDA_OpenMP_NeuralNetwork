/*
* Struct to define a neural network
*/
#ifndef MODEL_H
#define MODEL_H


#include <stdio.h>
#include <stdlib.h>
#include "layer.h"

typedef struct Model {
    int r; //the constant R received at compile time
    int size_nn; // total number of hidden layers without considering the input
    int size_input; //size of the input given to the model
    Layer* layers; //1D array of typedef layer
} Model;

/* Definition of the functions*/
Model new_model(int r, int size_nn, int size_input); //creates a new model
Layer* layers_init(int r, int size_nn, int size_input); //initializing the 1D array of layers of the model
double* predict(Model nn, double* input, int n_threads); // predict function for computing the output 
void free_model(Model nn); // deallocate a model's memory
#endif
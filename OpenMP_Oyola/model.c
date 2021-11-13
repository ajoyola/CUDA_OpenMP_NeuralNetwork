#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "model.h"
#include "layer.h"
#include <omp.h>

// Initializing the nn
Model new_model(int r, int size_nn, int size_input) {
	Model nn;
	nn.r = r;
	nn.size_nn = size_nn;
	nn.size_input = size_input;
	nn.layers = layers_init(r, size_nn, size_input); //initializing the hidden layers.
	return nn;
}

/*
layers_init: initializes all hidden layers in the model with random weights and random biases.
The output nodes are not computed here
input: r parameter of the model, size_nn as the number of layers, size_input as the number of nodes in the input
output: An 1D array of layers (each element is of typedef Layer)
*/
Layer* layers_init(int r, int size_nn, int size_input) {
	Layer* layers = (Layer*)malloc(size_nn * sizeof(Layer));
	for (int i = 0; i < size_nn; i++) {
		/* Computing the number of nodes for each layer using the formula: N - t( R-1)
		* here t = i+1 since the first layer is i=0*/
		int size_out = size_input - ((i + 1) * (r - 1));
		// random biases are generated inside the new_layer function
		layers[i] = new_layer(r, size_out);
	}
	return layers;
}

/* free_model: Deallocates the memory in the given model*/
void free_model(Model nn) {
	if (nn.size_nn > 0) {
		for (int i = 0; i < nn.size_nn; i++) free_layer(nn.layers[i]);
	}
	free(nn.layers); // free the array of layers
}

/*Predict: Computes the output of the model
* -Model nn to be computed and 1D array of double that contains input values
* - Output 1D array of double that contains the output of the model
* - n_threads is the number of threads with which the parallel regions will work.
*/

double* predict(Model nn, double* input, int n_threads) {
	double* output = (double*)malloc(nn.layers[0].n_neurons * sizeof(double)); // 1D array of output of the 1st layer
	double value = 0;
	for (int i = 0; i < nn.size_nn; i++) { // loop for layers
		// reallocating by each iteration since the output size will be reduced
		output = (double*)realloc(output, nn.layers[i].n_neurons * sizeof(double));
		int j;
		#pragma omp parallel for num_threads(n_threads) schedule(static,2)
		// loop for each output j in the layer i
		for (j = 0; j < nn.layers[i].n_neurons; j++) {
			int r;
			//#pragma omp parallel for num_threads(n_threads) schedule(static,2) reduction(+:value)
			// loop for the R relevant nodes of the output j
			for (r = 0; r < nn.r; r++) {
				value += input[j + r] * nn.layers[i].weights[j][r];
			}
			output[j] = activation(value + nn.layers[i].bias);
			value = 0;
		}
		double* tmp = input;
		input = output;
		output = tmp;
	}
	free(input);
	return output;
}





#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "layer.h"


/* new_layer: Initializes a single layer
Input: r parameter of the model, size_out is the number of output neurons in the layer
* previously computed in the model.
*/
Layer new_layer(int r, int size_out) {
	Layer l;
	l.bias = (double)rand() / (double)RAND_MAX; //generating random bias from 0 to 1
	l.weights = generate_rnd_weights(r, size_out); // 2D array of random weights
	l.n_neurons = size_out; // number of neurons, it is the length of the neurons array
	return l;
}

/* generate_rnd_weights: Generates a 2D vector with random values between -1 and 1
Input: R and the number of nodes in the output*/
double** generate_rnd_weights(int r, int size_out) {
	// allocating memory for the 2D matrix of weights
	double** weights = (double**)malloc(size_out * sizeof(double*));
	for (int i = 0; i < size_out; i++) {
		weights[i] = (double*)malloc(r * sizeof(double));
		// filling the rows with random values between 0 and 1
		for (int j = 0; j < r; j++) {
			weights[i][j] = ((double)rand() / ((double)RAND_MAX)) * 2 - 1; //random value from -1 and 1
		}
	}
	return weights;
}

// Sigmoid as the activation function of each layer
double activation(double x) { return 1.0 / (1.0 + exp(-x)); }

/* free_layer: Deallocates the memory previously allocated in the given layer*/
void free_layer(Layer layer) {
	free(layer.weights);
}


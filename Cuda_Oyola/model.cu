#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "model.cuh"
#include "layer.cuh"
#include "helper_functions.cuh"

#define BLKDIM 1024
#define R_size  3 // used by the static shared memory

// Initializing the nn
Model new_model(int r, int size_nn, int size_input) {
	Model nn;
	nn.r = r;
	nn.size_input = size_input;
	nn.size_nn = size_nn;
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
	/* For loop of layers*/
	for (int i = 0; i < size_nn; i++) {
		/* Computing the number of nodes for each layer using the formula: N - t( R-1)
		* here t = i+1 since the first layer is i=0*/
		int size_out = size_input - ((i + 1) * (r - 1));
		// random biases are generated inside the new_layer function
		layers[i] = new_layer(r, size_out);
	}
	return layers;
}

/* free_model: Deallocates the memory of the given model*/
void free_model(Model nn) {
	if (nn.size_nn > 0) {
		for (int i = 0; i < nn.size_nn; i++) free_layer(nn.layers[i]);
	}
	free(nn.layers); // free the array of layers
}

/* Calculate the throughput of the layer given the size of the output, the R constant and the elapsed tie
of the kernel in seconds*/
double get_throughput(int output_size, int r, float kernel_elapsedtime) {
	return 2 * output_size * r / (kernel_elapsedtime * pow(10, 6));
}

/* Kernel that works with the maximum number of threads and the minumun number of blocks as possible */
__global__ void maxthreads_compute_layer(double* weights, double bias, double* input, double* output, int input_size, int output_size, int input_per_block, int R, int sum_reduction)
{
	/* Working with dynamic shared memory for input and weights
	Each thread will load one element of the input and one of the 1D array of weights*/
	extern __shared__ double tmp_array[];

	/* Initializing useful variables and constants*/
	double value = 0;
	int offset = 0;
	const int t_id = threadIdx.x;
	const int g_weight_id = threadIdx.x + blockIdx.x * blockDim.x; //general weight index of the thread
	/* R threads will be grouped to calculate one output */
	const int n_output = floor(double(blockDim.x / R)); // number of outputs the block can compute
	const int l_output_id = floor(double(t_id / R)); // local output id that the thread belongs to 
	const int g_output_id = l_output_id + (blockIdx.x * blockDim.x / R);  // General output id
	const int t_r_id = t_id - R * l_output_id; // local index inside the output from 0 to R
	const int g_in_id = g_output_id + t_r_id; // general index of the corresponding input
	const int l_in_id = l_output_id + t_r_id; // local index of the corresponding input

	/* Shared memory tmp_array will be used for the input and the weights
	- tmp_input: 1D array of input
	- tmp_weights: 2D array of weights flattened into 1D, each value is accessed only by one single thread.
	*/
	double* tmp_input = (double*)&tmp_array;
	double* tmp_weights = (double*)&tmp_array[input_per_block]; // input_size as offset 

	/* Work only with the threads that compute the N output fo the block */
	if (g_output_id < output_size) {
		/* Read input into shared memory with the first N threads, where N is eqt the input's size computed by the block*/
		if (t_id < input_per_block) tmp_input[t_id] = input[g_in_id];
		/* Read weights into shared memory using one thread for each element*/
		tmp_weights[t_id] = weights[g_weight_id];
		/* Weighted input is a local computation done by each thread and saved into the shared memory*/
		tmp_weights[t_id] = tmp_input[l_in_id] * tmp_weights[t_id];
		/* Ensure that all threads have load their weighted input into the shared memory */
		__syncthreads();
	}

	/**********************************************************************************
	*                             OUTPUT COMPUTATION
	***********************************************************************************/
	/* Two approaches have been used*/
	//1) N threads to compute each one of the N output's elements
	if (sum_reduction == 0) {
		if (t_r_id == 0 and g_output_id < output_size) {
			for (offset = 0; offset < R; offset++) value += tmp_weights[t_id + offset];
			output[g_output_id] = 1.0 / (1.0 + exp(-(value + bias)));
		}
	}
	//2) R * N threads that compute the output's elements using sum reduction*/
	if (sum_reduction == 1) {
		if (g_output_id < output_size) {
			double b_size = R / 2;
			while (b_size >= 1) {
				if (t_r_id < floor(b_size)) {
					tmp_weights[t_id] += tmp_weights[t_id + (int)ceil(b_size)];
				}
				b_size = ceil(b_size) / 2;
				__syncthreads();
			}
			//Output is updated only by the first thread of each group//
			if (t_r_id == 0) {
				output[g_output_id] = 1.0 / (1.0 + exp(-(tmp_weights[t_id] + bias)));
			}
		}
	}
}


/* Kernel that works with R threads and N blocks as the output's size */
__global__ void rthreads_compute_layer(double* weights, double bias, double* input, double* output, int input_size, int R)
{
	/* Working with dynamic shared memory for input and weights
	Each thread will load one element of the input and one of the weights matrix*/
	//extern __shared__ double tmp_array[];
	__shared__ double tmp_array[2 * R_size];
	/* Initializing useful variables and constants*/
	double value = 0;
	int offset = 0;
	const int t_id = threadIdx.x;
	const int b_id = blockIdx.x;
	const int gi_id = threadIdx.x + blockIdx.x; // global id of the corresponding input 

	/* Shared memory tmp_array will be used for the input and the weights
	- tmp_input: 1D array, each element is accessed only by one single thread
	- tmp_weights: 1D array corresponding to the row of the current block id (output),
	  each element (column) is accessed only by one single thread.
	*/
	double* tmp_input = (double*)&tmp_array;
	double* tmp_weights = (double*)&tmp_array[R]; // R as offset 

	// global index should be as maximun the input's size - 1 
	if (gi_id < input_size) {
		/* Read input and weights into shared memory */
		tmp_input[t_id] = input[gi_id];
		tmp_weights[t_id] = weights[b_id * R + t_id];
		/* Weighted input is a local computation done by each thread and saved in the input's shared memory*/
		tmp_weights[t_id] = tmp_input[t_id] * tmp_weights[t_id];
		/* Ensure that all threads have load their weighted input into the shared memory */
		__syncthreads();
	}

	/**********************************************************************************
	*                             OUTPUT COMPUTATION
	***********************************************************************************/
	/* The output is computed only by the first thread*/
	if (t_id == 0) {
		for (offset = 0; offset < R; offset++) {
			value += tmp_weights[t_id + offset];
		}
		output[b_id] = 1.0 / (1.0 + exp(-(value + bias)));
	}


}

/*Predict: Kernel that compute the output of the model given the input,
* having the layers of the model with their biases and corresponding weights.
* Input: Model nn to be computed and 1D array of double that contains the input.
* Output: 1D array of double that contains the output of the model.
*/
double* predict(Model nn, double* input, char type[], double* t_tp) {

	/* Initialization of useful variables */
	// - Initializing the output according to the 1st layer
	double* output = (double*)malloc(nn.layers[0].n_neurons * sizeof(double));
	// - Size in bytes of the model's input
	int input_bytes = nn.size_input * sizeof(double);
	// - Size of the model's input, used to validate some operations inside the kernel
	int input_size = nn.size_input;
	// - Local throughput computed at each layer 
	double l_tp = 0;

	/* Loop for iterate the layers must be sequential */
	for (int i = 0; i < nn.size_nn; i++) {
		/* Initialization of useful variables */
		//  - Current layer
		Layer layer = nn.layers[i];
		//	- Size in bytes of the Output
		int output_bytes = nn.layers[i].n_neurons * sizeof(double);
		//	- Size of the Output 
		int output_size = nn.layers[i].n_neurons;
		//  - Reallocation of the output since it will be reduced along the for loop
		output = (double*)realloc(output, output_bytes);
		//  - 2D weights into 1D
		double* weights = (double*)malloc(output_size * nn.r * sizeof(double));
		for (int ii = 0; ii < output_size; ii++) {
			for (int jj = 0; jj < nn.r; jj++) {
				weights[ii * nn.r + jj] = layer.weights[ii][jj];
			}
		}
		// Instantiation of device copies: input, output, weights
		double* d_input, * d_output, * d_weights;
		/* Allocate space for device copies of weights, input, output*/
		CudaSafeCall(cudaMalloc((void**)&d_input, input_bytes));
		CudaSafeCall(cudaMalloc((void**)&d_output, output_bytes));
		CudaSafeCall(cudaMalloc((void**)&d_weights, output_bytes * nn.r));
		//CudaSafeCall(cudaMallocPitch(&d_weights, &pitch_d, width_bytes, layer.n_neurons));
		/* Copy inputs to device */
		CudaSafeCall(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(d_output, output, output_bytes, cudaMemcpyHostToDevice));
		CudaSafeCall(cudaMemcpy(d_weights, weights, output_bytes * nn.r, cudaMemcpyHostToDevice));
		/* Copy the 2D weights to device */
		//CudaSafeCall(cudaMemcpy2D(d_weights, pitch_d, layer.weights, width_bytes, width_bytes, layer.n_neurons, cudaMemcpyHostToDevice));

		// Measure time with cuda events
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		/* Launch compute_layer() kernel on GPU w
		   With the following configuration parameters:
			- GridSize:  Output's size of the current layer as number of blocks in the grid
			- BlockSize: R as number of threads for each block
			- SharedMemorySize: Size in bytes of twice R
		   And the following data parameters:
			- weights: 1D array of weights of the current layer
			- bias: Bias of the current layer
			- d_input: Input of the current layer
			- d_output: Output will be computer by the kernel and returnt to the host
			- input_size: Size of the input used for some validations and also as offset to work with more than one array in the shared memory
			- nn.R: R value defined at the beginning
			*/
		char s1[] = "maxthreads_sumreduc";
		char s2[] = "maxthreads_nonsumreduc";
		char s3[] = "rthreads";

		if (strcmp(type, s1) == 0 || strcmp(type, s2) == 0) {
			/* **********************************************************************************************
			*
			* Kernel that uses the maximun number of threads and the least number of blocks as possible
			*
			************************************************************************************************** */
			int sum_reduction = 0;
			if (strcmp(type, s1) == 0) sum_reduction = 1;
			int BlockSize = BLKDIM; //maximum number of threads per block
			/* Variables used by the Shared Memory*/
			int input_per_block = (floor((BLKDIM / nn.r)) + nn.r - 1); // part of the input used by threads in a block
			int inputbytes_per_block = input_per_block * sizeof(double); // part of the input used by threads in a block
			int SharedMemorySize = inputbytes_per_block + BLKDIM * sizeof(double); // 1D input + flattened 2D matrix of weights
			/*End*/
			if (nn.r * output_size < BLKDIM) {
				BlockSize = nn.r * output_size; // Each thread is responsible to calculate one weighted input
				SharedMemorySize = input_bytes + output_bytes * nn.r; // 1D input + flattened 2D matrix of weights
				input_per_block = input_size;
			}
			//int GridSize = (input_size + BlockSize - 1) / BlockSize;
			int GridSize = (output_size * nn.r) / BlockSize;
			cudaEventRecord(start);
			maxthreads_compute_layer << < GridSize, BlockSize, SharedMemorySize >> > (d_weights, layer.bias, d_input, d_output, input_size, output_size, input_per_block, nn.r, sum_reduction);
			cudaEventRecord(stop);
		}
		if (strcmp(type, s3) == 0) {
			/* ***********************************************************************************
			*
			* kernel that uses N blocks as output'size and R threads inside each block
			*
			*********************************************************************************** */
			int GridSize = output_size;
			int BlockSize = nn.r;

			cudaEventRecord(start);
			rthreads_compute_layer << < GridSize, BlockSize >> > (d_weights, layer.bias, d_input, d_output, input_size, nn.r);
			cudaEventRecord(stop);
		}
		cudaEventSynchronize(stop); //blocks CPU execution until the specified event is recorded
		CudaCheckError();

		// Compute elapsed time of the kernel
		float kernel_elapsedtime = 0;
		cudaEventElapsedTime(&kernel_elapsedtime, start, stop);
		// Compute and accumulate the throughput
		l_tp += get_throughput(output_size, nn.r, kernel_elapsedtime);

		/* Copy result of the input and output back to host */
		CudaSafeCall(cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost));
		CudaSafeCall(cudaMemcpy(input, d_input, output_bytes, cudaMemcpyDeviceToHost));

		// Setting the next input eqt the previous output
		double* tmp = input;
		input = output;
		output = tmp;
		input_bytes = output_bytes;
		input_size = nn.layers[i].n_neurons;

		/*Free device memory*/
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_weights);
		/*Free host memory*/
		free(weights);

	}
	/*Free host memory*/
	free(input);

	/*Average througput*/
	*t_tp = l_tp / nn.size_nn;

	/* Return the last output*/
	return output;
}

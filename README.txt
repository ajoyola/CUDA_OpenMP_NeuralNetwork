
Implementation of a Sparsely Connected Multi-layer Neural Network in C with OpenMP and CUDA

Course: Architectures and Platforms for Artificial Intelligence - UNIBO
Year: 2021 - 2021
Professor: Moreno Marzolla
Author: Angely Oyola S.
Id: 921487


The problem has been tackled with two versions, both described follow:

1) OpenMP
	This version uses OpenMP to compute a neural network with parallelism and the files can be found at: /OpenMP_Oyola

	1.1) Instructions to compile the program
	Run the below instructions in the command line. Set NAME with a string you desire.
	
		cd OpenMP_Oyola/
		gcc main.c model.c layer.c -lm  -lgomp -o NAME
		
	At this point the program has been compiled and saved in the current path with the given NAME.

	1.2) Instructions to run the program
	Run the below in the command line.
		./NAME N L P
	
	where:
	T: Number of threads.
	N: Size of the input.
	L: Number of layers in the model.
	P: Number of threads.
	
	Note: Given the kind of the neural network, it is needed the R value, which is a constant inside the program. If you want to change it, please go to:
		cd OpenMP_Oyola/main.c 
	and change the R value:
		line 7: #define R  3

2) CUDA
	This version uses CUDA to compute a neural network with parallelism and the files can be found at: /Cuda_Oyola

	2.1) Instructions to compile the program
	Run the below instructions in the command line. Set NAME with a string you desire.
	
		cd Cuda_Oyola/
		nvcc main.cu model.cu layer.cu -lm -o NAME
		
	At this point the program has been compiled and saved in the current path with the given NAME.

	2.2) Instructions to run the program
	Run the below in the command line.
	
		./NAME N L
	
	where:
	N: Size of the input.
	L: Number of layers in the model.

	Notes: 
	* Consider the same specification to the R constant as described in the section 1.2
	* Two different Kernels and three approaches have been built, the specifications of each one can be found in the report file.
	To choose the approach/kernel of preference you should go to:

		cd Cuda_Oyola/main.cu 
		Change the variable type to: maxthreads_nonsumreduc/ rthreads/ maxthreads_sumreduc 
		line 146: char type[] = "maxthreads_sumreduc";

	* The BKLDIM has been set to 1024, the maximun number of threads per block allowed by the GPU. 
	If you want to change this value, please consider your GPU's architecture and go to:
		cd Cuda_Oyola/model.cu 
		Change the BLKDIM value at line 10: #define BLKDIM 1024

The different strategies of the CUDA version have been tested on the lab machine whose technical details can be seen below:

CUDA capability: 2.0
Global memory: 3005 MB
CUDA cores: 512
Warp size: 32
Shared memory per block: 49152 B
Constant memory: 65536 B
Max number of threads per block: 1024
Max size of a thread block: (1024, 1024, 64)
Max grid size: (65535, 65535, 65535)


It is recommended to read the Report.pdf to find a deeper technical analysis of the project.

End
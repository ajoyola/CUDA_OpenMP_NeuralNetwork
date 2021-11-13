#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "model.h"

#define R  3 // number of neurons grouped to compute the output

/*clearscr: cleaning the screen in a portable way
* Source: [1]*/
void clearscr(void)
{
#ifdef _WIN32
    system("cls");
#elif defined(unix) || defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__))
    system("clear");
    //add some other OSes here if needed
#else
#error "OS not supported."
    //you can also throw an exception indicating the function can't be useds
#endif
}

/*
* generate_rnd_input: function to generate the input of the model
* with random values between 0 and 1, the size of the input is received as a parameter.
*/
double* generate_rnd_input(int size_input) {
    double* input = (double*)malloc(size_input * sizeof(double));
    for (int i = 0; i < size_input; i++) {
        input[i] = ((double)rand() / ((double)RAND_MAX * 2)) - 1; //random value from -1 to 1
    }
    return input;
}

//****************************************************************************
//****************************     MAIN      *********************************
//****************************************************************************

int main(int argc, char* argv[])
{
    /*Declaration of variables*/
    // size_input represents the size of the first layer and 
    // size_nn the number of hidden layers without considering the input
    int size_input = 0, size_nn = 0, n_threads = 0;
    char flag = 'N'; 

    //****************************     MENU      *********************************
    /*
    Since we expect to receive the values of size_input and size_nn as arguments, is valid
    to check if the 2 arguments were received */
    if (argc < 3) // Not arguments were received or less than 2
    {
        printf("The 2 arguments needed for running the program were not received!\n");
        while (flag != 'Y')
        {
            size_input = 0, size_nn = 0;
            while (size_input < R) {
                printf("\nInsert the size of the input, must be greater or equal than %d", R);
                printf(": ");
                scanf("%d", &size_input);
            }
            while (size_nn < 1) {
                printf("Insert the number of layers in the NN, must be at least 1: ");
                scanf("%d", &size_nn);
            }
            printf("\n\n*******************************************");
            printf("\n\nThe following values were received:\n");
            printf("The size of the input: %d\n", size_input);
            printf("The number of layers: %d\n", size_nn);
            printf("\nAre the values ok? [Y/N]: ");
            scanf("%c", &flag);
            clearscr();
        }
    }
    // With the 2 arguments already received, the NN can be built
    printf("\n\n\n*******************************************");
    printf("\n\n The NN is being built...");
    printf("\nWith the following arguments received ");
    if (flag != 'Y') //means the arguments were received by console, otherwise we use the ones received by keyboard
    {
        printf(" by console: \n");
        size_input = atoi(argv[1]);
        size_nn = atoi(argv[2]);
        n_threads = atoi(argv[3]); 
    }
    else { printf(" by keyboard: \n"); };
    printf("\n\tThe size of the 1st layer: %d\n", size_input);
    printf("\tThe number of total layers: %d\n", size_nn);
    printf("\tThe R value: %d\n", R);

    // Initializing the NN with the values received
    double* x = generate_rnd_input(size_input); // generating the input with random values
    Model nn = new_model(R, size_nn, size_input);

    printf("\tThe number of neurons in the output is: %d\n", nn.layers[size_nn - 1].n_neurons);

    /**********************************************************************************
    *    Generate the output of the model
    ***********************************************************************************/
    // Variables used for the performance anlysis of speedup
    double tstart, tstop;
    tstart = omp_get_wtime();
    // Call to the predict function
    double* y_pred = predict(nn, x, n_threads);
    tstop = omp_get_wtime();

    /**********************************************************************************
    *    Some prints
    ***********************************************************************************/

    // Uncoment the below part to see the generated weights, biases and input
    /*
    printf("\nValues of the Model\n");
    printf("\n\tInput of the Model\n");
    for (int i = 0; i < size_input; i++) {
        printf("\tThe neuron %d is %f\n", i, x[i]);
    }
    // Printing the Biases of the nn
    printf("\n\tBiases of the Model\n");
    for (int i = 0; i < size_nn; i++) {
        printf("\tThe Bias in the layer %d is %f\n", i, nn.layers[i].bias);
    }
    */

    // Printing the Output and the execution time in seconds
    printf("\n\n\tOutput of the Model\n");
    for (int i = 0; i < nn.layers[size_nn - 1].n_neurons; i++) {
        printf("\tThe Output %d out %d is %f\n", i, nn.layers[size_nn - 1].n_neurons - 1, y_pred[i]);
    }
    printf("\tExecution time: %f s\n", tstop - tstart);
    free_model(nn);
    return 0;




}
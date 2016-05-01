#ifndef LSTMLAYER_H
#define LSTMLAYER_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#include "text.h"
#include "lstmlayerstate.h"

using namespace std;

class LSTMLayer // A LSTM layer in time (can contain sub-layers)
{
public:
    uint32_t layerStateArrayPos;
    uint32_t layerStateArraySize;
    LSTMLayerState **layerStates; // Stores previous iterations

    double learningRate;
    uint32_t inputCount;
    uint32_t outputCount;
    uint32_t backpropagationSteps;
    uint32_t hiddenLayerCount;
    uint32_t totalLayerCount;
    uint32_t *hiddenLayerNeuronCounts;

    static double sig(double input); // sigmoid function
    static double tanh(double input); // tanh function

    static double *cloneDoubleArray(double *array,uint32_t size);
    static double *mergeDoubleArrays(double *array1,uint32_t size1,double *array2,uint32_t size2);
    static double *multiplyDoubleArrayByDoubleArray(double *array1,uint32_t size,double *array2);
    static double *multiplyDoubleArray(double *array,uint32_t size,double factor);
    static double *addToDoubleArray(double *array,uint32_t size,double summand);
    static double sumDoubleArray(double *array,uint32_t size);
    static void directlyMultiplyDoubleArrayByDoubleArray(double *array1,uint32_t size,double *array2);
    static void directlyMultiplyDoubleArray(double *array,uint32_t size,double factor);
    static void directlyAddToDoubleArray(double *array,uint32_t size,double summand);
    static void fillDoubleArray(double *array,uint32_t size,double value);
    static void fillDoubleArrayWithRandomValues(double *array,uint32_t size,double from,double to);

    LSTMLayerState *pushLayerState();
    LSTMLayerState *getCurrentLayerState();
    bool hasLayerState(uint32_t stepsBack);
    uint32_t getAvailableStepsBack();
    LSTMLayerState *getLayerState(uint32_t stepsBack);

    // Please note that the cell count is equal to the output count!
    // To have more cells than outputs (essential in most situations, as it makes the network more powerful), you should use the first n required output values only!
    LSTMLayer(uint32_t _inputCount,uint32_t _outputCount,uint32_t _backpropagationSteps,double _learningRate,uint32_t hiddenLayerCount=0,uint32_t *_hiddenLayerNeuronCounts=0);
    ~LSTMLayer();

    double *process(double *input);
    // Takes in the desired outputs of the last n=backpropagationSteps states and the current state, beginning with the oldest state and ending with the current state.
    void learn(double **desiredOutputs);
};

#endif // LSTMLAYER_H

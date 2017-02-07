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
#include "lstmstate.h"

using namespace std;

class LSTM
{
public:
    uint32_t stateArrayPos;
    uint32_t stateArraySize;
    LSTMState **states; // Stores previous iterations

    // Dimensions: Layers - neurons in this layer - weights from neurons in previous layer to neurons in this layer
    double ***previousForgetGateWeightDeltas;
    double ***previousInputGateWeightDeltas;
    double ***previousOutputGateWeightDeltas;
    double ***previousCandidateGateWeightDeltas;
    // Dimensions: Layers - neurons in this layer
    double **previousForgetGateBiasWeightDeltas;
    double **previousInputGateBiasWeightDeltas;
    double **previousOutputGateBiasWeightDeltas;
    double **previousCandidateGateBiasWeightDeltas;
    // Dimensions: Cells
    double *previousForgetGateValueSumBiasWeightDeltas;
    double *previousInputGateValueSumBiasWeightDeltas;
    double *previousOutputGateValueSumBiasWeightDeltas;
    double *previousCandidateGateValueSumBiasWeightDeltas;

    double learningRate;
    double momentum;
    double weightDecay;
    double forgetGateNetworkLearningRate;
    double inputGateNetworkLearningRate;
    double outputGateNetworkLearningRate;
    double candidateGateNetworkLearningRate;
    double forgetGateNetworkMomentum;
    double inputGateNetworkMomentum;
    double outputGateNetworkMomentum;
    double candidateGateNetworkMomentum;
    double forgetGateNetworkWeightDecay;
    double inputGateNetworkWeightDecay;
    double outputGateNetworkWeightDecay;
    double candidateGateNetworkWeightDecay;
    uint32_t inputCount;
    uint32_t outputCount;
    uint32_t backpropagationSteps;
    uint32_t forgetGateHiddenLayerCount;
    uint32_t inputGateHiddenLayerCount;
    uint32_t outputGateHiddenLayerCount;
    uint32_t candidateGateHiddenLayerCount;
    uint32_t forgetGateTotalLayerCount;
    uint32_t inputGateTotalLayerCount;
    uint32_t outputGateTotalLayerCount;
    uint32_t candidateGateTotalLayerCount;
    uint32_t *forgetGateHiddenLayerNeuronCounts;
    uint32_t *inputGateHiddenLayerNeuronCounts;
    uint32_t *outputGateHiddenLayerNeuronCounts;
    uint32_t *candidateGateHiddenLayerNeuronCounts;

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

    LSTMState *pushState();
    LSTMState *getCurrentState();
    bool hasState(uint32_t stepsBack);
    uint32_t getAvailableStepsBack();
    LSTMState *getState(uint32_t stepsBack);

    // Please note that the cell count is equal to the output count!
    // To have more cells than outputs (essential in most situations, as it makes the network more powerful), you should use the first n required output values only!
    LSTM(uint32_t _inputCount,uint32_t _outputCount,uint32_t _backpropagationSteps,double _learningRate,double _momentum,double _weightDecay,double _networkLearningRate=std::numeric_limits<double>::min(),double _networkMomentum=std::numeric_limits<double>::min(),double _networkWeightDecay=std::numeric_limits<double>::min(),uint32_t _forgetGateHiddenLayerCount=0,uint32_t *_forgetGateHiddenLayerNeuronCounts=0,uint32_t _inputGateHiddenLayerCount=0,uint32_t *_inputGateHiddenLayerNeuronCounts=0,uint32_t _outputGateHiddenLayerCount=0,uint32_t *_outputGateHiddenLayerNeuronCounts=0,uint32_t _candidateGateHiddenLayerCount=0,uint32_t *_candidateGateHiddenLayerNeuronCounts=0);
    ~LSTM();

    double *process(double *input);
    // Takes in the desired outputs of the last n=backpropagationSteps states and the current state, beginning with the oldest state and ending with the current state.
    void learn(double **desiredOutputs);
};

#endif // LSTMLAYER_H

#ifndef LSTMSTATE_H
#define LSTMSTATE_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
#include <math.h>
#include <time.h>

class LSTMState
{
public:
    // Dimensions: Cells - layers - neurons in this layer - weights from neurons in previous layer to neurons in this layer
    double ****forgetGateLayerWeights;
    double ****inputGateLayerWeights;
    double ****outputGateLayerWeights;
    double ****candidateGateLayerWeights;
    // Dimensions: Cells - layers - inputs/outputs
    double ***forgetGateLayerBiasWeights;
    double ***inputGateLayerBiasWeights;
    double ***outputGateLayerBiasWeights;
    double ***candidateGateLayerBiasWeights;
    // Dimensions: Cells - inputs/outputs (final weights)
    double **forgetGatePreValues;
    double **inputGatePreValues;
    double **outputGatePreValues;
    double **candidateGatePreValues;
    // Dimensions: Cells - layers - neuron values
    double ***forgetGateLayerNeuronValues;
    double ***inputGateLayerNeuronValues;
    double ***outputGateLayerNeuronValues;
    double ***candidateGateLayerNeuronValues;
    // Dimensions: Cells
    double *forgetGateValues;
    double *inputGateValues;
    double *outputGateValues;
    double *candidateGateValues;
    double *forgetGateValueSumBiasWeights;
    double *inputGateValueSumBiasWeights;
    double *outputGateValueSumBiasWeights;
    double *candidateGateValueSumBiasWeights;
    double *input;
    double *output;
    double *desiredOutput;
    double *cellStates;

    uint32_t inputCount;
    uint32_t outputCount;
    uint32_t inputAndOutputCount;

    uint32_t forgetGateTotalLayerCount;
    uint32_t inputGateTotalLayerCount;
    uint32_t outputGateTotalLayerCount;
    uint32_t candidateGateTotalLayerCount;

    uint32_t *forgetGateHiddenLayerNeuronCounts;
    uint32_t *inputGateHiddenLayerNeuronCounts;
    uint32_t *outputGateHiddenLayerNeuronCounts;
    uint32_t *candidateGateHiddenLayerNeuronCounts;

    double *bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastCellStates; // bottom_diff_s
    double *bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastOutputs; // bottom_diff_h
    double *bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToInputs; // bottom_diff_x

    static double sig(double input); // sigmoid function
    static double tanh(double input); // tanh function

    LSTMState(LSTMState *copyFrom=0,uint32_t _inputCount=0,uint32_t _outputCount=0,uint32_t _forgetGateHiddenLayerCount=0,uint32_t *_forgetGateHiddenLayerNeuronCounts=0,uint32_t _inputGateHiddenLayerCount=0,uint32_t *_inputGateHiddenLayerNeuronCounts=0,uint32_t _outputGateHiddenLayerCount=0,uint32_t *_outputGateHiddenLayerNeuronCounts=0,uint32_t _candidateGateHiddenLayerCount=0,uint32_t *_candidateGateHiddenLayerNeuronCounts=0);
    void calculateGatePreValues(double *previousOutputs); // Takes "input" and calculates the values to be multiplied by the weights of the gates (in single-layer LSTM: inputGateWeights[cell][i]*input[i]; here: inputGatePreValues[cell][i]).
    void freeMemory();
    ~LSTMState();
};

#endif // LSTMSTATE_H

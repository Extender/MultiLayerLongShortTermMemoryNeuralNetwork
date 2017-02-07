#include "lstmstate.h"

double LSTMState::sig(double input)
{
    // Derivative: sig(input)*(1.0-sig(input))
    return 1.0/(1.0+pow(M_E,-input));
}

double LSTMState::tanh(double input)
{
    // Derivative: 1.0-pow(tanh(input),2.0)
    return (1.0-pow(M_E,-2.0*input))/(1.0+pow(M_E,-2.0*input));
}

LSTMState::LSTMState(LSTMState *copyFrom, uint32_t _inputCount, uint32_t _outputCount, uint32_t _forgetGateHiddenLayerCount, uint32_t *_forgetGateHiddenLayerNeuronCounts, uint32_t _inputGateHiddenLayerCount, uint32_t *_inputGateHiddenLayerNeuronCounts, uint32_t _outputGateHiddenLayerCount, uint32_t *_outputGateHiddenLayerNeuronCounts, uint32_t _candidateGateHiddenLayerCount, uint32_t *_candidateGateHiddenLayerNeuronCounts)
{
    bool copy=copyFrom!=0;
    inputCount=copy?copyFrom->inputCount:_inputCount;
    outputCount=copy?copyFrom->outputCount:_outputCount;
    forgetGateTotalLayerCount=copy?copyFrom->forgetGateTotalLayerCount:_forgetGateHiddenLayerCount+1/*Topmost output layer*/;
    inputGateTotalLayerCount=copy?copyFrom->inputGateTotalLayerCount:_inputGateHiddenLayerCount+1/*Topmost output layer*/;
    outputGateTotalLayerCount=copy?copyFrom->outputGateTotalLayerCount:_outputGateHiddenLayerCount+1/*Topmost output layer*/;
    candidateGateTotalLayerCount=copy?copyFrom->candidateGateTotalLayerCount:_candidateGateHiddenLayerCount+1/*Topmost output layer*/;
    uint32_t forgetGateTotalLayerBasedDoublePointerArraySize=forgetGateTotalLayerCount*sizeof(double*);
    uint32_t inputGateTotalLayerBasedDoublePointerArraySize=inputGateTotalLayerCount*sizeof(double*);
    uint32_t outputGateTotalLayerBasedDoublePointerArraySize=outputGateTotalLayerCount*sizeof(double*);
    uint32_t candidateGateTotalLayerBasedDoublePointerArraySize=candidateGateTotalLayerCount*sizeof(double*);
    uint32_t forgetGateHiddenLayerCountBasedArraySize=(copy?copyFrom->forgetGateTotalLayerCount-1:_forgetGateHiddenLayerCount)*sizeof(uint32_t);
    uint32_t inputGateHiddenLayerCountBasedArraySize=(copy?copyFrom->inputGateTotalLayerCount-1:_inputGateHiddenLayerCount)*sizeof(uint32_t);
    uint32_t outputGateHiddenLayerCountBasedArraySize=(copy?copyFrom->outputGateTotalLayerCount-1:_outputGateHiddenLayerCount)*sizeof(uint32_t);
    uint32_t candidateGateHiddenLayerCountBasedArraySize=(copy?copyFrom->candidateGateTotalLayerCount-1:_candidateGateHiddenLayerCount)*sizeof(uint32_t);
    forgetGateHiddenLayerNeuronCounts=(uint32_t*)malloc(forgetGateHiddenLayerCountBasedArraySize);
    inputGateHiddenLayerNeuronCounts=(uint32_t*)malloc(inputGateHiddenLayerCountBasedArraySize);
    outputGateHiddenLayerNeuronCounts=(uint32_t*)malloc(outputGateHiddenLayerCountBasedArraySize);
    candidateGateHiddenLayerNeuronCounts=(uint32_t*)malloc(candidateGateHiddenLayerCountBasedArraySize);
    memcpy(forgetGateHiddenLayerNeuronCounts,copy?copyFrom->forgetGateHiddenLayerNeuronCounts:_forgetGateHiddenLayerNeuronCounts,forgetGateHiddenLayerCountBasedArraySize);
    memcpy(inputGateHiddenLayerNeuronCounts,copy?copyFrom->inputGateHiddenLayerNeuronCounts:_inputGateHiddenLayerNeuronCounts,inputGateHiddenLayerCountBasedArraySize);
    memcpy(outputGateHiddenLayerNeuronCounts,copy?copyFrom->outputGateHiddenLayerNeuronCounts:_outputGateHiddenLayerNeuronCounts,outputGateHiddenLayerCountBasedArraySize);
    memcpy(candidateGateHiddenLayerNeuronCounts,copy?copyFrom->candidateGateHiddenLayerNeuronCounts:_candidateGateHiddenLayerNeuronCounts,candidateGateHiddenLayerCountBasedArraySize);

    inputAndOutputCount=inputCount+outputCount;

    uint32_t outputBasedDoubleArraySize=outputCount*sizeof(double);
    uint32_t outputBasedDoublePointerArraySize=outputCount*sizeof(double*);
    uint32_t outputBasedDoublePointerPointerArraySize=outputCount*sizeof(double**); // Will be the same as outputBasedDoublePointerArraySize.
    uint32_t outputBasedDoublePointerPointerPointerArraySize=outputCount*sizeof(double***); // Will be the same as outputBasedDoublePointerArraySize.
    uint32_t inputAndOutputBasedDoubleArraySize=inputAndOutputCount*sizeof(double);
    forgetGatePreValues=(double**)malloc(outputBasedDoublePointerArraySize);
    inputGatePreValues=(double**)malloc(outputBasedDoublePointerArraySize);
    outputGatePreValues=(double**)malloc(outputBasedDoublePointerArraySize);
    candidateGatePreValues=(double**)malloc(outputBasedDoublePointerArraySize);
    forgetGateLayerBiasWeights=(double***)malloc(outputBasedDoublePointerPointerArraySize); // First dimension: cells
    inputGateLayerBiasWeights=(double***)malloc(outputBasedDoublePointerPointerArraySize); // First dimension: cells
    outputGateLayerBiasWeights=(double***)malloc(outputBasedDoublePointerPointerArraySize); // First dimension: cells
    candidateGateLayerBiasWeights=(double***)malloc(outputBasedDoublePointerPointerArraySize); // First dimension: cells
    forgetGateLayerNeuronValues=(double***)malloc(outputBasedDoublePointerPointerArraySize); // First dimension: cells
    inputGateLayerNeuronValues=(double***)malloc(outputBasedDoublePointerPointerArraySize); // First dimension: cells
    outputGateLayerNeuronValues=(double***)malloc(outputBasedDoublePointerPointerArraySize); // First dimension: cells
    candidateGateLayerNeuronValues=(double***)malloc(outputBasedDoublePointerPointerArraySize); // First dimension: cells
    forgetGateLayerWeights=(double****)malloc(outputBasedDoublePointerPointerPointerArraySize); // First dimension: cells
    inputGateLayerWeights=(double****)malloc(outputBasedDoublePointerPointerPointerArraySize); // First dimension: cells
    outputGateLayerWeights=(double****)malloc(outputBasedDoublePointerPointerPointerArraySize); // First dimension: cells
    candidateGateLayerWeights=(double****)malloc(outputBasedDoublePointerPointerPointerArraySize); // First dimension: cells
    input=(double*)malloc(inputCount*sizeof(double));
    output=(double*)malloc(outputBasedDoubleArraySize);
    desiredOutput=(double*)malloc(outputBasedDoubleArraySize);
    cellStates=(double*)malloc(outputBasedDoubleArraySize);
    forgetGateValues=(double*)malloc(outputBasedDoubleArraySize);
    inputGateValues=(double*)malloc(outputBasedDoubleArraySize);
    outputGateValues=(double*)malloc(outputBasedDoubleArraySize);
    candidateGateValues=(double*)malloc(outputBasedDoubleArraySize);
    forgetGateValueSumBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    inputGateValueSumBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    outputGateValueSumBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    candidateGateValueSumBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    // The derivatives do not need to be initialized.
    bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastCellStates=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_s
    bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastOutputs=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_h
    bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToInputs=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_x
    if(copyFrom==0)
    {
        srand((uint32_t)time(0));
        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            // First dimension: cells

            forgetGateValueSumBiasWeights[cell]=0.0;
            inputGateValueSumBiasWeights[cell]=0.0;
            outputGateValueSumBiasWeights[cell]=0.0;
            candidateGateValueSumBiasWeights[cell]=0.0;

            forgetGateLayerBiasWeights[cell]=(double**)malloc(forgetGateTotalLayerBasedDoublePointerArraySize);
            inputGateLayerBiasWeights[cell]=(double**)malloc(inputGateTotalLayerBasedDoublePointerArraySize);
            outputGateLayerBiasWeights[cell]=(double**)malloc(outputGateTotalLayerBasedDoublePointerArraySize);
            candidateGateLayerBiasWeights[cell]=(double**)malloc(candidateGateTotalLayerBasedDoublePointerArraySize);

            forgetGateLayerNeuronValues[cell]=(double**)malloc(forgetGateTotalLayerBasedDoublePointerArraySize);
            inputGateLayerNeuronValues[cell]=(double**)malloc(inputGateTotalLayerBasedDoublePointerArraySize);
            outputGateLayerNeuronValues[cell]=(double**)malloc(outputGateTotalLayerBasedDoublePointerArraySize);
            candidateGateLayerNeuronValues[cell]=(double**)malloc(candidateGateTotalLayerBasedDoublePointerArraySize);

            forgetGateLayerWeights[cell]=(double***)malloc(forgetGateTotalLayerBasedDoublePointerArraySize);
            inputGateLayerWeights[cell]=(double***)malloc(inputGateTotalLayerBasedDoublePointerArraySize);
            outputGateLayerWeights[cell]=(double***)malloc(outputGateTotalLayerBasedDoublePointerArraySize);
            candidateGateLayerWeights[cell]=(double***)malloc(candidateGateTotalLayerBasedDoublePointerArraySize);


            uint32_t neuronsInLastLayer=inputAndOutputCount; // First layer: inputs and previous outputs

            // Forget gate
            for(uint32_t thisLayer=0;thisLayer<forgetGateTotalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==forgetGateTotalLayerCount-1?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                forgetGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                forgetGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                forgetGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: neurons in this layer

                    forgetGateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);

                    // The neuron values do not need to be initialized.

                    uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                    forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);

                    for(uint32_t neuronInLastLayer=0;neuronInLastLayer<neuronsInLastLayer;neuronInLastLayer++)
                    {
                        // Next dimension: weights from neurons in previous layer to neurons in this layer
                        forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer][neuronInLastLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                    }
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }

            // Input gate
            for(uint32_t thisLayer=0;thisLayer<inputGateTotalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==inputGateTotalLayerCount-1?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                inputGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                inputGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                inputGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: neurons in this layer

                    inputGateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);

                    // The neuron values do not need to be initialized.

                    uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                    inputGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);

                    for(uint32_t neuronInLastLayer=0;neuronInLastLayer<neuronsInLastLayer;neuronInLastLayer++)
                    {
                        // Next dimension: weights from neurons in previous layer to neurons in this layer
                        inputGateLayerWeights[cell][thisLayer][neuronInThisLayer][neuronInLastLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                    }
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }

            // Output gate
            for(uint32_t thisLayer=0;thisLayer<outputGateTotalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==outputGateTotalLayerCount-1?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                outputGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                outputGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                outputGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: neurons in this layer

                    outputGateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);

                    // The neuron values do not need to be initialized.

                    uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                    outputGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);

                    for(uint32_t neuronInLastLayer=0;neuronInLastLayer<neuronsInLastLayer;neuronInLastLayer++)
                    {
                        // Next dimension: weights from neurons in previous layer to neurons in this layer
                        outputGateLayerWeights[cell][thisLayer][neuronInThisLayer][neuronInLastLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                    }
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }

            // Candidate gate
            for(uint32_t thisLayer=0;thisLayer<candidateGateTotalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==candidateGateTotalLayerCount-1?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                candidateGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                candidateGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                candidateGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: neurons in this layer

                    candidateGateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);

                    // The neuron values do not need to be initialized.

                    uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                    candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);

                    for(uint32_t neuronInLastLayer=0;neuronInLastLayer<neuronsInLastLayer;neuronInLastLayer++)
                    {
                        // Next dimension: weights from neurons in previous layer to neurons in this layer
                        candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer][neuronInLastLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                    }
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }

            // These 4 arrays do not need to be initialized yet:
            forgetGatePreValues[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            inputGatePreValues[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            outputGatePreValues[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            candidateGatePreValues[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
        }
    }
    else
    {
        memcpy(forgetGateValueSumBiasWeights,copyFrom->forgetGateValueSumBiasWeights,outputBasedDoubleArraySize);
        memcpy(inputGateValueSumBiasWeights,copyFrom->forgetGateValueSumBiasWeights,outputBasedDoubleArraySize);
        memcpy(outputGateValueSumBiasWeights,copyFrom->forgetGateValueSumBiasWeights,outputBasedDoubleArraySize);
        memcpy(candidateGateValueSumBiasWeights,copyFrom->forgetGateValueSumBiasWeights,outputBasedDoubleArraySize);

        // Create deep copies of the two-dimensional weight arrays, the three-dimensional layer bias weight arrays and the four-dimensional layer weight arrays:
        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            // First dimension: cells

            forgetGateLayerBiasWeights[cell]=(double**)malloc(forgetGateTotalLayerBasedDoublePointerArraySize);
            inputGateLayerBiasWeights[cell]=(double**)malloc(inputGateTotalLayerBasedDoublePointerArraySize);
            outputGateLayerBiasWeights[cell]=(double**)malloc(outputGateTotalLayerBasedDoublePointerArraySize);
            candidateGateLayerBiasWeights[cell]=(double**)malloc(candidateGateTotalLayerBasedDoublePointerArraySize);

            forgetGateLayerNeuronValues[cell]=(double**)malloc(forgetGateTotalLayerBasedDoublePointerArraySize);
            inputGateLayerNeuronValues[cell]=(double**)malloc(inputGateTotalLayerBasedDoublePointerArraySize);
            outputGateLayerNeuronValues[cell]=(double**)malloc(outputGateTotalLayerBasedDoublePointerArraySize);
            candidateGateLayerNeuronValues[cell]=(double**)malloc(candidateGateTotalLayerBasedDoublePointerArraySize);

            forgetGateLayerWeights[cell]=(double***)malloc(forgetGateTotalLayerBasedDoublePointerArraySize);
            inputGateLayerWeights[cell]=(double***)malloc(inputGateTotalLayerBasedDoublePointerArraySize);
            outputGateLayerWeights[cell]=(double***)malloc(outputGateTotalLayerBasedDoublePointerArraySize);
            candidateGateLayerWeights[cell]=(double***)malloc(candidateGateTotalLayerBasedDoublePointerArraySize);

            uint32_t neuronsInLastLayer=inputAndOutputCount; // First layer: inputs and previous outputs

            // Forget gate
            for(uint32_t thisLayer=0;thisLayer<forgetGateTotalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==forgetGateTotalLayerCount-1?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                forgetGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                forgetGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                forgetGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                // Next dimension: neurons in this layer

                memcpy(forgetGateLayerBiasWeights[cell][thisLayer],copyFrom->forgetGateLayerBiasWeights[cell][thisLayer],neuronsInThisLayerBasedDoubleArraySize);

                // The neuron values do not need to be initialized.

                uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: weights from neurons in previous layer to neurons in this layer

                    forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    memcpy(forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer],copyFrom->forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer],neuronsInLastLayerBasedDoubleArraySize);
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }

            // Input gate
            for(uint32_t thisLayer=0;thisLayer<inputGateTotalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==inputGateTotalLayerCount-1?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                inputGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                inputGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                inputGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                // Next dimension: neurons in this layer

                memcpy(inputGateLayerBiasWeights[cell][thisLayer],copyFrom->inputGateLayerBiasWeights[cell][thisLayer],neuronsInThisLayerBasedDoubleArraySize);

                // The neuron values do not need to be initialized.

                uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: weights from neurons in previous layer to neurons in this layer

                    inputGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    memcpy(inputGateLayerWeights[cell][thisLayer][neuronInThisLayer],copyFrom->inputGateLayerWeights[cell][thisLayer][neuronInThisLayer],neuronsInLastLayerBasedDoubleArraySize);
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }

            // Output gate
            for(uint32_t thisLayer=0;thisLayer<outputGateTotalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==outputGateTotalLayerCount-1?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                outputGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                outputGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                outputGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                // Next dimension: neurons in this layer

                memcpy(outputGateLayerBiasWeights[cell][thisLayer],copyFrom->outputGateLayerBiasWeights[cell][thisLayer],neuronsInThisLayerBasedDoubleArraySize);

                // The neuron values do not need to be initialized.

                uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: weights from neurons in previous layer to neurons in this layer

                    outputGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    memcpy(outputGateLayerWeights[cell][thisLayer][neuronInThisLayer],copyFrom->outputGateLayerWeights[cell][thisLayer][neuronInThisLayer],neuronsInLastLayerBasedDoubleArraySize);
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }

            // Candidate gate
            for(uint32_t thisLayer=0;thisLayer<candidateGateTotalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==candidateGateTotalLayerCount-1?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                candidateGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                candidateGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                candidateGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                // Next dimension: neurons in this layer

                memcpy(candidateGateLayerBiasWeights[cell][thisLayer],copyFrom->candidateGateLayerBiasWeights[cell][thisLayer],neuronsInThisLayerBasedDoubleArraySize);

                // The neuron values do not need to be initialized.

                uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: weights from neurons in previous layer to neurons in this layer

                    candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    memcpy(candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer],copyFrom->candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer],neuronsInLastLayerBasedDoubleArraySize);
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }

            // These 4 arrays do not need to be initialized yet:
            forgetGatePreValues[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            inputGatePreValues[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            outputGatePreValues[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            candidateGatePreValues[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
        }
    }
}

void LSTMState::calculateGatePreValues(double *previousOutputs)
{
    // Inputs used: "input"; previous outputs used: "previousOutputs"
    // First layer: inputs and previous outputs
    // [hidden layers]
    // Last layer: output (size: size of inputs + previous outputs); to be used in lstmlayer.cpp.

    // (Basic multilayer feedforward neural network principle)

    // One MLFFNNT for each cell's forget, input, output and candidate gates.

    // For each gate

    double ****gateLayerWeights;
    double ***gateLayerBiasWeights;
    double **gatePreValues;
    double **gateNeuronValues;
    uint32_t gateTotalLayerCount;
    uint32_t *gateHiddenLayerNeuronCounts;

    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        uint32_t neuronsInLastLayer;

        for(uint8_t gate=1;gate<=4;gate++)
        {
            if(gate==1)
            {
                // Forget gate
                gateLayerWeights=forgetGateLayerWeights;
                gateLayerBiasWeights=forgetGateLayerBiasWeights;
                gatePreValues=forgetGatePreValues;
                gateNeuronValues=forgetGateLayerNeuronValues[cell];
                gateTotalLayerCount=forgetGateTotalLayerCount;
                gateHiddenLayerNeuronCounts=forgetGateHiddenLayerNeuronCounts;
            }
            else if(gate==2)
            {
                // Input gate
                gateLayerWeights=inputGateLayerWeights;
                gateLayerBiasWeights=inputGateLayerBiasWeights;
                gatePreValues=inputGatePreValues;
                gateNeuronValues=inputGateLayerNeuronValues[cell];
                gateTotalLayerCount=inputGateTotalLayerCount;
                gateHiddenLayerNeuronCounts=inputGateHiddenLayerNeuronCounts;
            }
            else if(gate==3)
            {
                // Output gate
                gateLayerWeights=outputGateLayerWeights;
                gateLayerBiasWeights=outputGateLayerBiasWeights;
                gatePreValues=outputGatePreValues;
                gateNeuronValues=outputGateLayerNeuronValues[cell];
                gateTotalLayerCount=outputGateTotalLayerCount;
                gateHiddenLayerNeuronCounts=outputGateHiddenLayerNeuronCounts;
            }
            else // if(gate==4)
            {
                // Candidate gate
                gateLayerWeights=candidateGateLayerWeights;
                gateLayerBiasWeights=candidateGateLayerBiasWeights;
                gatePreValues=candidateGatePreValues;
                gateNeuronValues=candidateGateLayerNeuronValues[cell];
                gateTotalLayerCount=candidateGateTotalLayerCount;
                gateHiddenLayerNeuronCounts=candidateGateHiddenLayerNeuronCounts;
            }

            neuronsInLastLayer=inputAndOutputCount;
            for(uint32_t thisLayer=0;thisLayer<gateTotalLayerCount/*Topmost output layer included*/;thisLayer++)
            {
                uint32_t neuronsInThisLayer=thisLayer==gateTotalLayerCount-1?inputAndOutputCount:gateHiddenLayerNeuronCounts[thisLayer];
                // Get previous layer's values, multiply by weights, add biases, and put the output through the tanh function.

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    double inputsTimesWeightsSum=0.0;
                    if(thisLayer==0)
                    {
                        // Use input/previous output values
                        for(uint32_t inputN=0;inputN<inputCount;inputN++)
                            inputsTimesWeightsSum+=input[inputN]*gateLayerWeights[cell][thisLayer][neuronInThisLayer][inputN];
                        if(previousOutputs!=0)
                        {
                            for(uint32_t outputN=0;outputN<outputCount;outputN++)
                                inputsTimesWeightsSum+=previousOutputs[outputN]*gateLayerWeights[cell][thisLayer][neuronInThisLayer][inputCount+outputN];
                        }
                    }
                    else
                    {
                        for(uint32_t neuronInLastLayer=0;neuronInLastLayer<neuronsInLastLayer;neuronInLastLayer++)
                            inputsTimesWeightsSum+=gateNeuronValues[thisLayer-1][neuronInLastLayer]*gateLayerWeights[cell][thisLayer][neuronInThisLayer][neuronInLastLayer];
                    }
                    gateNeuronValues[thisLayer][neuronInThisLayer]=tanh(inputsTimesWeightsSum+gateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]);
                }

                neuronsInLastLayer=neuronsInThisLayer;
            }
            // Copy values of topmost layer into pre-value array
            memcpy(gatePreValues[cell],gateNeuronValues[gateTotalLayerCount-1/*The topmost layer which outputs the values into the gate pre-value array*/],inputAndOutputCount*sizeof(double));
        }
    }
}

void LSTMState::freeMemory()
{
    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        // Forget gate
        for(uint32_t thisLayer=0;thisLayer<forgetGateTotalLayerCount;thisLayer++)
        {
            // Free layer bias weights
            free(forgetGateLayerBiasWeights[cell][thisLayer]);
            free(forgetGateLayerNeuronValues[cell][thisLayer]);
            // Free weights from neurons in previous layer to neurons in this layer
            uint32_t neuronsInThisLayer=thisLayer==forgetGateTotalLayerCount-1?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[thisLayer];
            for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                free(forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer]);
            free(forgetGateLayerWeights[cell][thisLayer]);
        }

        // Input gate
        for(uint32_t thisLayer=0;thisLayer<inputGateTotalLayerCount;thisLayer++)
        {
            // Free layer bias weights
            free(inputGateLayerBiasWeights[cell][thisLayer]);
            free(inputGateLayerNeuronValues[cell][thisLayer]);
            // Free weights from neurons in previous layer to neurons in this layer
            uint32_t neuronsInThisLayer=thisLayer==inputGateTotalLayerCount-1?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[thisLayer];
            for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                free(inputGateLayerWeights[cell][thisLayer][neuronInThisLayer]);
            free(inputGateLayerWeights[cell][thisLayer]);
        }

        // Output gate
        for(uint32_t thisLayer=0;thisLayer<outputGateTotalLayerCount;thisLayer++)
        {
            // Free layer bias weights
            free(outputGateLayerBiasWeights[cell][thisLayer]);
            free(outputGateLayerNeuronValues[cell][thisLayer]);
            // Free weights from neurons in previous layer to neurons in this layer
            uint32_t neuronsInThisLayer=thisLayer==outputGateTotalLayerCount-1?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[thisLayer];
            for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                free(outputGateLayerWeights[cell][thisLayer][neuronInThisLayer]);
            free(outputGateLayerWeights[cell][thisLayer]);
        }

        // Candidate gate
        for(uint32_t thisLayer=0;thisLayer<candidateGateTotalLayerCount;thisLayer++)
        {
            // Free layer bias weights
            free(candidateGateLayerBiasWeights[cell][thisLayer]);
            free(candidateGateLayerNeuronValues[cell][thisLayer]);
            // Free weights from neurons in previous layer to neurons in this layer
            uint32_t neuronsInThisLayer=thisLayer==candidateGateTotalLayerCount-1?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[thisLayer];
            for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                free(candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer]);
            free(candidateGateLayerWeights[cell][thisLayer]);
        }

        free(forgetGateLayerBiasWeights[cell]);
        free(inputGateLayerBiasWeights[cell]);
        free(outputGateLayerBiasWeights[cell]);
        free(candidateGateLayerBiasWeights[cell]);
        free(forgetGateLayerNeuronValues[cell]);
        free(inputGateLayerNeuronValues[cell]);
        free(outputGateLayerNeuronValues[cell]);
        free(candidateGateLayerNeuronValues[cell]);
        free(forgetGateLayerWeights[cell]);
        free(inputGateLayerWeights[cell]);
        free(outputGateLayerWeights[cell]);
        free(candidateGateLayerWeights[cell]);
        free(forgetGatePreValues[cell]);
        free(inputGatePreValues[cell]);
        free(outputGatePreValues[cell]);
        free(candidateGatePreValues[cell]);
    }
    free(input);
    free(output);
    free(desiredOutput);
    free(cellStates);
    free(forgetGateLayerBiasWeights);
    free(inputGateLayerBiasWeights);
    free(outputGateLayerBiasWeights);
    free(candidateGateLayerBiasWeights);
    free(forgetGateLayerNeuronValues);
    free(inputGateLayerNeuronValues);
    free(outputGateLayerNeuronValues);
    free(candidateGateLayerNeuronValues);
    free(forgetGatePreValues);
    free(inputGatePreValues);
    free(outputGatePreValues);
    free(candidateGatePreValues);
    free(forgetGateValues);
    free(inputGateValues);
    free(outputGateValues);
    free(candidateGateValues);
    free(forgetGateValueSumBiasWeights);
    free(inputGateValueSumBiasWeights);
    free(outputGateValueSumBiasWeights);
    free(candidateGateValueSumBiasWeights);
    free(forgetGateLayerWeights);
    free(inputGateLayerWeights);
    free(outputGateLayerWeights);
    free(candidateGateLayerWeights);
    free(bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastCellStates);
    free(bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToInputs);
    free(bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastOutputs);
    free(forgetGateHiddenLayerNeuronCounts);
    free(inputGateHiddenLayerNeuronCounts);
    free(outputGateHiddenLayerNeuronCounts);
    free(candidateGateHiddenLayerNeuronCounts);
}

LSTMState::~LSTMState()
{
    freeMemory();
}

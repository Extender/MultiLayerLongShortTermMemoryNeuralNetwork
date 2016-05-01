#include "lstmlayerstate.h"

double LSTMLayerState::sig(double input)
{
    // Derivative: sig(input)*(1.0-sig(input))
    return 1.0/(1.0+pow(M_E,-input));
}

double LSTMLayerState::tanh(double input)
{
    // Derivative: 1.0-((tanh(input))^2)
    return (1.0-pow(M_E,-2.0*input))/(1.0+pow(M_E,-2.0*input));
}

LSTMLayerState::LSTMLayerState(LSTMLayerState *copyFrom,uint32_t _inputCount,uint32_t _outputCount,uint32_t _hiddenLayerCount,uint32_t *_hiddenLayerNeuronCounts)
{
    bool copy=copyFrom!=0;
    inputCount=copy?copyFrom->inputCount:_inputCount;
    outputCount=copy?copyFrom->outputCount:_outputCount;
    totalLayerCount=copy?copyFrom->totalLayerCount:_hiddenLayerCount+1/*Topmost output layer*/;
    uint32_t totalLayerCountBasedArraySize=totalLayerCount*sizeof(uint32_t);
    uint32_t hiddenLayerCountBasedArraySize=(copy?copyFrom->totalLayerCount-1:_hiddenLayerCount)*sizeof(uint32_t);
    hiddenLayerNeuronCounts=(uint32_t*)malloc(hiddenLayerCountBasedArraySize);
    memcpy(hiddenLayerNeuronCounts,copy?copyFrom->hiddenLayerNeuronCounts:_hiddenLayerNeuronCounts,hiddenLayerCountBasedArraySize);
    inputAndOutputCount=inputCount+outputCount;

    uint32_t outputBasedDoubleArraySize=outputCount*sizeof(double);
    uint32_t outputBasedDoublePointerArraySize=outputCount*sizeof(double*);
    uint32_t outputBasedDoublePointerPointerArraySize=outputCount*sizeof(double**); // Will be the same as outputBasedDoublePointerArraySize.
    uint32_t outputBasedDoublePointerPointerPointerArraySize=outputCount*sizeof(double***); // Will be the same as outputBasedDoublePointerArraySize.
    uint32_t inputAndOutputBasedDoubleArraySize=inputAndOutputCount*sizeof(double);
    uint32_t totalLayerBasedDoublePointerArraySize=totalLayerCount*sizeof(double*);
    uint32_t totalLayerBasedDoublePointerPointerArraySize=totalLayerCount*sizeof(double**);
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
    forgetGateBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    inputGateBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    outputGateBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    candidateGateBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    // The derivatives do not need to be initialized.
    bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToCellStates=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_s
    bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToOutputs=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_h
    bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToInputs=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_x
    if(copyFrom==0)
    {
        srand(time(NULL));
        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            // First dimension: cells

            forgetGateBiasWeights[cell]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
            inputGateBiasWeights[cell]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
            outputGateBiasWeights[cell]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
            candidateGateBiasWeights[cell]=-0.1+0.2*((double)rand()/(double)RAND_MAX);

            forgetGateLayerBiasWeights[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            inputGateLayerBiasWeights[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            outputGateLayerBiasWeights[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            candidateGateLayerBiasWeights[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);

            forgetGateLayerNeuronValues[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            inputGateLayerNeuronValues[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            outputGateLayerNeuronValues[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            candidateGateLayerNeuronValues[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);

            forgetGateLayerWeights[cell]=(double***)malloc(totalLayerBasedDoublePointerPointerArraySize);
            inputGateLayerWeights[cell]=(double***)malloc(totalLayerBasedDoublePointerPointerArraySize);
            outputGateLayerWeights[cell]=(double***)malloc(totalLayerBasedDoublePointerPointerArraySize);
            candidateGateLayerWeights[cell]=(double***)malloc(totalLayerBasedDoublePointerPointerArraySize);


            uint32_t neuronsInLastLayer=inputAndOutputCount; // First layer: inputs and previous outputs

            for(uint32_t thisLayer=0;thisLayer<totalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==totalLayerCount-1?inputAndOutputCount:hiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                forgetGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                inputGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                outputGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                candidateGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);

                forgetGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                inputGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                outputGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                candidateGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);


                forgetGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);
                inputGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);
                outputGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);
                candidateGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: neurons in this layer

                    forgetGateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                    inputGateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                    outputGateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                    candidateGateLayerBiasWeights[cell][thisLayer][neuronInThisLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);

                    // The neuron values do not need to be initialized.

                    uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                    forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    inputGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    outputGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);

                    for(uint32_t neuronInLastLayer=0;neuronInLastLayer<neuronsInLastLayer;neuronInLastLayer++)
                    {
                        // Next dimension: weights from neurons in previous layer to neurons in this layer
                        forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer][neuronInLastLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                        inputGateLayerWeights[cell][thisLayer][neuronInThisLayer][neuronInLastLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                        outputGateLayerWeights[cell][thisLayer][neuronInThisLayer][neuronInLastLayer]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
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
        memcpy(forgetGateBiasWeights,copyFrom->forgetGateBiasWeights,outputBasedDoubleArraySize);
        memcpy(inputGateBiasWeights,copyFrom->forgetGateBiasWeights,outputBasedDoubleArraySize);
        memcpy(outputGateBiasWeights,copyFrom->forgetGateBiasWeights,outputBasedDoubleArraySize);
        memcpy(candidateGateBiasWeights,copyFrom->forgetGateBiasWeights,outputBasedDoubleArraySize);

        // Create deep copies of the two-dimensional weight arrays, the three-dimensional layer bias weight arrays and the four-dimensional layer weight arrays:
        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            // First dimension: cells

            forgetGateLayerBiasWeights[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            inputGateLayerBiasWeights[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            outputGateLayerBiasWeights[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            candidateGateLayerBiasWeights[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);

            forgetGateLayerNeuronValues[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            inputGateLayerNeuronValues[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            outputGateLayerNeuronValues[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);
            candidateGateLayerNeuronValues[cell]=(double**)malloc(totalLayerBasedDoublePointerArraySize);

            forgetGateLayerWeights[cell]=(double***)malloc(totalLayerBasedDoublePointerPointerArraySize);
            inputGateLayerWeights[cell]=(double***)malloc(totalLayerBasedDoublePointerPointerArraySize);
            outputGateLayerWeights[cell]=(double***)malloc(totalLayerBasedDoublePointerPointerArraySize);
            candidateGateLayerWeights[cell]=(double***)malloc(totalLayerBasedDoublePointerPointerArraySize);

            uint32_t neuronsInLastLayer=inputAndOutputCount; // First layer: inputs and previous outputs

            for(uint32_t thisLayer=0;thisLayer<totalLayerCount;thisLayer++)
            {
                // Next dimension: layers

                uint32_t neuronsInThisLayer=thisLayer==totalLayerCount-1?inputAndOutputCount:hiddenLayerNeuronCounts[thisLayer];
                uint32_t neuronsInThisLayerBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
                uint32_t neuronsInThisLayerBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
                forgetGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                inputGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                outputGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                candidateGateLayerBiasWeights[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);

                forgetGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                inputGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                outputGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);
                candidateGateLayerNeuronValues[cell][thisLayer]=(double*)malloc(neuronsInThisLayerBasedDoubleArraySize);

                forgetGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);
                inputGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);
                outputGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);
                candidateGateLayerWeights[cell][thisLayer]=(double**)malloc(neuronsInThisLayerBasedDoublePointerArraySize);

                // Next dimension: neurons in this layer

                memcpy(forgetGateLayerBiasWeights[cell][thisLayer],copyFrom->forgetGateLayerBiasWeights[cell][thisLayer],neuronsInThisLayerBasedDoubleArraySize);
                memcpy(inputGateLayerBiasWeights[cell][thisLayer],copyFrom->inputGateLayerBiasWeights[cell][thisLayer],neuronsInThisLayerBasedDoubleArraySize);
                memcpy(outputGateLayerBiasWeights[cell][thisLayer],copyFrom->outputGateLayerBiasWeights[cell][thisLayer],neuronsInThisLayerBasedDoubleArraySize);
                memcpy(candidateGateLayerBiasWeights[cell][thisLayer],copyFrom->candidateGateLayerBiasWeights[cell][thisLayer],neuronsInThisLayerBasedDoubleArraySize);

                // The neuron values do not need to be initialized.

                uint32_t neuronsInLastLayerBasedDoubleArraySize=neuronsInLastLayer*sizeof(double);

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Next dimension: weights from neurons in previous layer to neurons in this layer

                    forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    inputGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    outputGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);
                    candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer]=(double*)malloc(neuronsInLastLayerBasedDoubleArraySize);

                    memcpy(forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer],copyFrom->forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer],neuronsInLastLayerBasedDoubleArraySize);
                    memcpy(inputGateLayerWeights[cell][thisLayer][neuronInThisLayer],copyFrom->inputGateLayerWeights[cell][thisLayer][neuronInThisLayer],neuronsInLastLayerBasedDoubleArraySize);
                    memcpy(outputGateLayerWeights[cell][thisLayer][neuronInThisLayer],copyFrom->outputGateLayerWeights[cell][thisLayer][neuronInThisLayer],neuronsInLastLayerBasedDoubleArraySize);
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

void LSTMLayerState::calculateGatePreValues(double *previousOutputs)
{
    // Inputs used: "input"; previous outputs used: "previousOutputs"
    // First layer: inputs and previous outputs
    // [hidden layers]
    // Last layer: output (size: size of inputs + previous outputs); to be used in lstmlayer.cpp.

    // (Basic multilayer feedforward neural network principle)

    // One MLFFNNT for each cell's forget, input, output and candidate gates.


    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        uint32_t neuronsInLastLayer;

        for(uint8_t gate=1;gate<=4;gate++)
        {
            // For each gate

            double ****gateLayerWeights;
            double ***gateLayerBiasWeights;
            double **gatePreValues;
            double **gateNeuronValues;
            if(gate==1)
            {
                // Forget gate
                gateLayerWeights=forgetGateLayerWeights;
                gateLayerBiasWeights=forgetGateLayerBiasWeights;
                gatePreValues=forgetGatePreValues;
                gateNeuronValues=forgetGateLayerNeuronValues[cell];
            }
            else if(gate==2)
            {
                // Input gate
                gateLayerWeights=inputGateLayerWeights;
                gateLayerBiasWeights=inputGateLayerBiasWeights;
                gatePreValues=inputGatePreValues;
                gateNeuronValues=inputGateLayerNeuronValues[cell];
            }
            else if(gate==3)
            {
                // Output gate
                gateLayerWeights=outputGateLayerWeights;
                gateLayerBiasWeights=outputGateLayerBiasWeights;
                gatePreValues=outputGatePreValues;
                gateNeuronValues=outputGateLayerNeuronValues[cell];
            }
            else // if(gate==4)
            {
                // Candidate gate
                gateLayerWeights=candidateGateLayerWeights;
                gateLayerBiasWeights=candidateGateLayerBiasWeights;
                gatePreValues=candidateGatePreValues;
                gateNeuronValues=candidateGateLayerNeuronValues[cell];
            }

            neuronsInLastLayer=inputAndOutputCount;
            for(uint32_t thisLayer=0;thisLayer<totalLayerCount/*Topmost output layer included*/;thisLayer++)
            {
                uint32_t neuronsInThisLayer=thisLayer==totalLayerCount-1?inputAndOutputCount:hiddenLayerNeuronCounts[thisLayer];
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
            memcpy(gatePreValues[cell],gateNeuronValues[totalLayerCount-1/*The topmost layer which outputs the values into the gate pre-value array*/],inputAndOutputCount*sizeof(double));
        }
    }
}

void LSTMLayerState::freeMemory()
{
    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        for(uint32_t thisLayer=0;thisLayer<totalLayerCount;thisLayer++)
        {
            // Free layer bias weights
            free(forgetGateLayerBiasWeights[cell][thisLayer]);
            free(inputGateLayerBiasWeights[cell][thisLayer]);
            free(outputGateLayerBiasWeights[cell][thisLayer]);
            free(candidateGateLayerBiasWeights[cell][thisLayer]);
            free(forgetGateLayerNeuronValues[cell][thisLayer]);
            free(inputGateLayerNeuronValues[cell][thisLayer]);
            free(outputGateLayerNeuronValues[cell][thisLayer]);
            free(candidateGateLayerNeuronValues[cell][thisLayer]);
            // Free weights from neurons in previous layer to neurons in this layer
            uint32_t neuronsInThisLayer=thisLayer==totalLayerCount-1?inputAndOutputCount:hiddenLayerNeuronCounts[thisLayer];
            for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
            {
                free(forgetGateLayerWeights[cell][thisLayer][neuronInThisLayer]);
                free(inputGateLayerWeights[cell][thisLayer][neuronInThisLayer]);
                free(outputGateLayerWeights[cell][thisLayer][neuronInThisLayer]);
                free(candidateGateLayerWeights[cell][thisLayer][neuronInThisLayer]);
            }
            free(forgetGateLayerWeights[cell][thisLayer]);
            free(inputGateLayerWeights[cell][thisLayer]);
            free(outputGateLayerWeights[cell][thisLayer]);
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
    free(forgetGateBiasWeights);
    free(inputGateBiasWeights);
    free(outputGateBiasWeights);
    free(candidateGateBiasWeights);
    free(forgetGateLayerWeights);
    free(inputGateLayerWeights);
    free(outputGateLayerWeights);
    free(candidateGateLayerWeights);
    free(bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToCellStates);
    free(bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToInputs);
    free(bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToOutputs);
    free(hiddenLayerNeuronCounts);
}

LSTMLayerState::~LSTMLayerState()
{
    freeMemory();
}

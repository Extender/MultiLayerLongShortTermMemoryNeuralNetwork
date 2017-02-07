#include "lstm.h"

double LSTM::sig(double input)
{
    // Derivative: sig(input)*(1.0-sig(input))
    return 1.0/(1.0+pow(M_E,-input));
}

double LSTM::tanh(double input)
{
    // Derivative: 1.0-pow(tanh(input),2.0)
    return (1.0-pow(M_E,-2.0*input))/(1.0+pow(M_E,-2.0*input));
}

double *LSTM::cloneDoubleArray(double *array, uint32_t size)
{
    size_t arraySize=size*sizeof(double);
    double *out=(double*)malloc(arraySize);
    memcpy(out,array,arraySize);
    return out;
}

double *LSTM::mergeDoubleArrays(double *array1, uint32_t size1, double *array2, uint32_t size2)
{
    size_t array1Size=size1*sizeof(double);
    size_t array2Size=size2*sizeof(double);
    size_t arraySize=size1+size2;
    double *out=(double*)malloc(arraySize);
    memcpy(out,array1,array1Size);
    memcpy(out+array1Size,array2,array2Size);
    return out;
}

double *LSTM::multiplyDoubleArrayByDoubleArray(double *array1, uint32_t size, double *array2)
{
    double *out=cloneDoubleArray(array1,size);
    for(uint32_t i=0;i<size;i++)
        out[i]*=array2[i];
    return out;
}

double *LSTM::multiplyDoubleArray(double *array, uint32_t size, double factor)
{
    double *out=cloneDoubleArray(array,size);
    for(uint32_t i=0;i<size;i++)
        out[i]*=factor;
    return out;
}

double *LSTM::addToDoubleArray(double *array, uint32_t size, double summand)
{
    double *out=cloneDoubleArray(array,size);
    for(uint32_t i=0;i<size;i++)
        out[i]+=summand;
    return out;
}

double LSTM::sumDoubleArray(double *array, uint32_t size)
{
    double out=0.0;
    for(uint32_t i=0;i<size;i++)
        out+=array[i];
    return out;
}

void LSTM::directlyMultiplyDoubleArrayByDoubleArray(double *array1, uint32_t size, double *array2)
{
    for(uint32_t i=0;i<size;i++)
        array1[i]*=array2[i];
}

void LSTM::directlyMultiplyDoubleArray(double *array, uint32_t size, double factor)
{
    for(uint32_t i=0;i<size;i++)
        array[i]*=factor;
}

void LSTM::directlyAddToDoubleArray(double *array, uint32_t size, double summand)
{
    for(uint32_t i=0;i<size;i++)
        array[i]+=summand;
}

void LSTM::fillDoubleArray(double *array, uint32_t size, double value)
{
    for(uint32_t i=0;i<size;i++)
        array[i]=value;
}

void LSTM::fillDoubleArrayWithRandomValues(double *array, uint32_t size, double from, double to)
{
    srand((uint32_t)time(0));
    for(uint32_t i=0;i<size;i++)
        array[i]=from+((double)rand()/(double)RAND_MAX)*(to-from);
}

LSTMState *LSTM::pushState()
{
    // This works as follows: the buffer is larger (usually 2 times larger) than the required size, allowing us to avoid having to move memory
    // every time a new state is pushed. Once the buffer is filled, the needed elements in the front are moved back, overriding the old states
    // that aren't needed anymore, and creating room for new states to be pushed.

    if(stateArrayPos==0xffffffff)
        stateArrayPos=0; // Do not increment the position the first time pushState() is called.
    else
    {
        if(stateArrayPos==stateArraySize-1)
        {
            // Overwrite old states that aren't needed anymore, and set the new position:
            // Note that the current state will be a backpropagation state after the new state is pushed to the array.
            delete states[stateArrayPos-backpropagationSteps]; // Delete unneeded state
            memcpy(states,states+(stateArraySize-backpropagationSteps),backpropagationSteps*sizeof(LSTMState*));
            stateArrayPos=backpropagationSteps-1;
        }
        stateArrayPos++;
    }
    // Copy values from previous state, if such a state exists:
    LSTMState *newState=stateArrayPos>0/*Has previous state?*/?new LSTMState(getState(1)):new LSTMState(0,inputCount,outputCount,forgetGateHiddenLayerCount,forgetGateHiddenLayerNeuronCounts,inputGateHiddenLayerCount,inputGateHiddenLayerNeuronCounts,outputGateHiddenLayerCount,outputGateHiddenLayerNeuronCounts,candidateGateHiddenLayerCount,candidateGateHiddenLayerNeuronCounts);
    states[stateArrayPos]=newState;
    if(stateArrayPos>backpropagationSteps)
    {
        // Free memory occupied by the now unneeded state (each time a new state is pushed, the memory occupied by the oldest state, which is
        // not needed anymore from that point on, is freed):
        delete states[stateArrayPos-backpropagationSteps-1];
    }
    return states[stateArrayPos];
}

LSTMState *LSTM::getCurrentState()
{
    return states[stateArrayPos];
}

bool LSTM::hasState(uint32_t stepsBack)
{
    return stateArrayPos!=0xffffffff&&stepsBack<=__min(backpropagationSteps,stateArrayPos);
}

uint32_t LSTM::getAvailableStepsBack()
{
    return stateArrayPos!=0xffffffff?__min(backpropagationSteps,stateArrayPos):0;
}

LSTMState *LSTM::getState(uint32_t stepsBack)
{
    return states[stateArrayPos-stepsBack];
}

LSTM::LSTM(uint32_t _inputCount, uint32_t _outputCount, uint32_t _backpropagationSteps, double _learningRate, double _momentum, double _weightDecay, double _networkLearningRate, double _networkMomentum, double _networkWeightDecay, uint32_t _forgetGateHiddenLayerCount, uint32_t *_forgetGateHiddenLayerNeuronCounts, uint32_t _inputGateHiddenLayerCount, uint32_t *_inputGateHiddenLayerNeuronCounts, uint32_t _outputGateHiddenLayerCount, uint32_t *_outputGateHiddenLayerNeuronCounts, uint32_t _candidateGateHiddenLayerCount, uint32_t *_candidateGateHiddenLayerNeuronCounts)
{
    inputCount=_inputCount;
    outputCount=_outputCount;
    uint32_t inputAndOutputCount=inputCount+outputCount;
    backpropagationSteps=_backpropagationSteps;
    learningRate=_learningRate;
    momentum=_momentum;
    weightDecay=_weightDecay;

    if(_networkLearningRate==std::numeric_limits<double>::min())
        _networkLearningRate=_learningRate;
    if(_networkMomentum==std::numeric_limits<double>::min())
        _networkMomentum=_momentum;
    if(_networkWeightDecay==std::numeric_limits<double>::min())
        _networkWeightDecay=_weightDecay;

    forgetGateNetworkLearningRate=_networkLearningRate;
    inputGateNetworkLearningRate=_networkLearningRate;
    outputGateNetworkLearningRate=_networkLearningRate;
    candidateGateNetworkLearningRate=_networkLearningRate;
    forgetGateNetworkMomentum=_networkMomentum;
    inputGateNetworkMomentum=_networkMomentum;
    outputGateNetworkMomentum=_networkMomentum;
    candidateGateNetworkMomentum=_networkMomentum;
    forgetGateNetworkWeightDecay=_networkWeightDecay;
    inputGateNetworkWeightDecay=_networkWeightDecay;
    outputGateNetworkWeightDecay=_networkWeightDecay;
    candidateGateNetworkWeightDecay=_networkWeightDecay;

    stateArraySize=2*backpropagationSteps+1 /*One for the current state.*/;
    stateArrayPos=0xffffffff;
    states=(LSTMState**)malloc(stateArraySize*sizeof(LSTMState*));

    forgetGateHiddenLayerCount=_forgetGateHiddenLayerCount;
    inputGateHiddenLayerCount=_inputGateHiddenLayerCount;
    outputGateHiddenLayerCount=_outputGateHiddenLayerCount;
    candidateGateHiddenLayerCount=_candidateGateHiddenLayerCount;

    size_t outputCountBasedDoubleArraySize=outputCount*sizeof(double);
    previousForgetGateValueSumBiasWeightDeltas=(double*)malloc(outputCountBasedDoubleArraySize);
    previousInputGateValueSumBiasWeightDeltas=(double*)malloc(outputCountBasedDoubleArraySize);
    previousOutputGateValueSumBiasWeightDeltas=(double*)malloc(outputCountBasedDoubleArraySize);
    previousCandidateGateValueSumBiasWeightDeltas=(double*)malloc(outputCountBasedDoubleArraySize);

    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        previousForgetGateValueSumBiasWeightDeltas[cell]=0.0;
        previousInputGateValueSumBiasWeightDeltas[cell]=0.0;
        previousOutputGateValueSumBiasWeightDeltas[cell]=0.0;
        previousCandidateGateValueSumBiasWeightDeltas[cell]=0.0;
    }

    // Copy hidden layer neuron counts (to avoid errors)

    // Forget gate
    uint32_t forgetGateHiddenLayerNeuronCountArraySize=forgetGateHiddenLayerCount*sizeof(uint32_t);
    forgetGateHiddenLayerNeuronCounts=(uint32_t*)malloc(forgetGateHiddenLayerNeuronCountArraySize);
    if(_forgetGateHiddenLayerNeuronCounts==0)
    {
        for(uint32_t hiddenLayer=0;hiddenLayer<forgetGateHiddenLayerCount;hiddenLayer++)
            forgetGateHiddenLayerNeuronCounts[hiddenLayer]=inputAndOutputCount;
    }
    else
        memcpy(forgetGateHiddenLayerNeuronCounts,_forgetGateHiddenLayerNeuronCounts,forgetGateHiddenLayerNeuronCountArraySize);

    // Input gate
    uint32_t inputGateHiddenLayerNeuronCountArraySize=inputGateHiddenLayerCount*sizeof(uint32_t);
    inputGateHiddenLayerNeuronCounts=(uint32_t*)malloc(inputGateHiddenLayerNeuronCountArraySize);
    if(_inputGateHiddenLayerNeuronCounts==0)
    {
        for(uint32_t hiddenLayer=0;hiddenLayer<inputGateHiddenLayerCount;hiddenLayer++)
            inputGateHiddenLayerNeuronCounts[hiddenLayer]=inputAndOutputCount;
    }
    else
        memcpy(inputGateHiddenLayerNeuronCounts,_inputGateHiddenLayerNeuronCounts,inputGateHiddenLayerNeuronCountArraySize);

    // Output gate
    uint32_t outputGateHiddenLayerNeuronCountArraySize=outputGateHiddenLayerCount*sizeof(uint32_t);
    outputGateHiddenLayerNeuronCounts=(uint32_t*)malloc(outputGateHiddenLayerNeuronCountArraySize);
    if(_outputGateHiddenLayerNeuronCounts==0)
    {
        for(uint32_t hiddenLayer=0;hiddenLayer<outputGateHiddenLayerCount;hiddenLayer++)
            outputGateHiddenLayerNeuronCounts[hiddenLayer]=inputAndOutputCount;
    }
    else
        memcpy(outputGateHiddenLayerNeuronCounts,_outputGateHiddenLayerNeuronCounts,outputGateHiddenLayerNeuronCountArraySize);

    // Candidate gate
    uint32_t candidateGateHiddenLayerNeuronCountArraySize=candidateGateHiddenLayerCount*sizeof(uint32_t);
    candidateGateHiddenLayerNeuronCounts=(uint32_t*)malloc(candidateGateHiddenLayerNeuronCountArraySize);
    if(_candidateGateHiddenLayerNeuronCounts==0)
    {
        for(uint32_t hiddenLayer=0;hiddenLayer<candidateGateHiddenLayerCount;hiddenLayer++)
            candidateGateHiddenLayerNeuronCounts[hiddenLayer]=inputAndOutputCount;
    }
    else
        memcpy(candidateGateHiddenLayerNeuronCounts,_candidateGateHiddenLayerNeuronCounts,candidateGateHiddenLayerNeuronCountArraySize);

    forgetGateTotalLayerCount=_forgetGateHiddenLayerCount+1;
    inputGateTotalLayerCount=_inputGateHiddenLayerCount+1;
    outputGateTotalLayerCount=_outputGateHiddenLayerCount+1;
    candidateGateTotalLayerCount=_candidateGateHiddenLayerCount+1;

    size_t forgetGateTotalLayerCountBasedDoublePointerArraySize=forgetGateTotalLayerCount*sizeof(double*);
    size_t inputGateTotalLayerCountBasedDoublePointerArraySize=inputGateTotalLayerCount*sizeof(double*);
    size_t outputGateTotalLayerCountBasedDoublePointerArraySize=outputGateTotalLayerCount*sizeof(double*);
    size_t candidateGateTotalLayerCountBasedDoublePointerArraySize=candidateGateTotalLayerCount*sizeof(double*);

    previousForgetGateBiasWeightDeltas=(double**)malloc(forgetGateTotalLayerCountBasedDoublePointerArraySize);
    previousInputGateBiasWeightDeltas=(double**)malloc(inputGateTotalLayerCountBasedDoublePointerArraySize);
    previousOutputGateBiasWeightDeltas=(double**)malloc(outputGateTotalLayerCountBasedDoublePointerArraySize);
    previousCandidateGateBiasWeightDeltas=(double**)malloc(candidateGateTotalLayerCountBasedDoublePointerArraySize);
    previousForgetGateWeightDeltas=(double***)malloc(forgetGateTotalLayerCountBasedDoublePointerArraySize);
    previousInputGateWeightDeltas=(double***)malloc(inputGateTotalLayerCountBasedDoublePointerArraySize);
    previousOutputGateWeightDeltas=(double***)malloc(outputGateTotalLayerCountBasedDoublePointerArraySize);
    previousCandidateGateWeightDeltas=(double***)malloc(candidateGateTotalLayerCountBasedDoublePointerArraySize);

    // Forget gate
    for(uint32_t currentLayer=0;currentLayer<forgetGateTotalLayerCount;currentLayer++)
    {
        uint32_t neuronsInThisLayer=currentLayer==forgetGateTotalLayerCount-1?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[currentLayer];
        uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[currentLayer-1];
        size_t thisLayerNeuronCountBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
        size_t thisLayerNeuronCountBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
        size_t previousLayerNeuronCountBasedDoubleArraySize=neuronsInPreviousLayer*sizeof(double);
        previousForgetGateBiasWeightDeltas[currentLayer]=(double*)malloc(thisLayerNeuronCountBasedDoubleArraySize);
        previousForgetGateWeightDeltas[currentLayer]=(double**)malloc(thisLayerNeuronCountBasedDoublePointerArraySize);

        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
        {
            previousForgetGateBiasWeightDeltas[currentLayer][neuronInThisLayer]=0.0;
            previousForgetGateWeightDeltas[currentLayer][neuronInThisLayer]=(double*)malloc(previousLayerNeuronCountBasedDoubleArraySize);
            for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                previousForgetGateWeightDeltas[currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
        }
    }

    // Input gate
    for(uint32_t currentLayer=0;currentLayer<inputGateTotalLayerCount;currentLayer++)
    {
        uint32_t neuronsInThisLayer=currentLayer==inputGateTotalLayerCount-1?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[currentLayer];
        uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[currentLayer-1];
        size_t thisLayerNeuronCountBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
        size_t thisLayerNeuronCountBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
        size_t previousLayerNeuronCountBasedDoubleArraySize=neuronsInPreviousLayer*sizeof(double);
        previousInputGateBiasWeightDeltas[currentLayer]=(double*)malloc(thisLayerNeuronCountBasedDoubleArraySize);
        previousInputGateWeightDeltas[currentLayer]=(double**)malloc(thisLayerNeuronCountBasedDoublePointerArraySize);

        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
        {
            previousInputGateBiasWeightDeltas[currentLayer][neuronInThisLayer]=0.0;
            previousInputGateWeightDeltas[currentLayer][neuronInThisLayer]=(double*)malloc(previousLayerNeuronCountBasedDoubleArraySize);
            for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                previousInputGateWeightDeltas[currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
        }
    }

    // Output gate
    for(uint32_t currentLayer=0;currentLayer<outputGateTotalLayerCount;currentLayer++)
    {
        uint32_t neuronsInThisLayer=currentLayer==outputGateTotalLayerCount-1?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[currentLayer];
        uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[currentLayer-1];
        size_t thisLayerNeuronCountBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
        size_t thisLayerNeuronCountBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
        size_t previousLayerNeuronCountBasedDoubleArraySize=neuronsInPreviousLayer*sizeof(double);
        previousOutputGateBiasWeightDeltas[currentLayer]=(double*)malloc(thisLayerNeuronCountBasedDoubleArraySize);
        previousOutputGateWeightDeltas[currentLayer]=(double**)malloc(thisLayerNeuronCountBasedDoublePointerArraySize);

        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
        {
            previousOutputGateBiasWeightDeltas[currentLayer][neuronInThisLayer]=0.0;
            previousOutputGateWeightDeltas[currentLayer][neuronInThisLayer]=(double*)malloc(previousLayerNeuronCountBasedDoubleArraySize);
            for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                previousOutputGateWeightDeltas[currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
        }
    }

    // Candidate gate
    for(uint32_t currentLayer=0;currentLayer<candidateGateTotalLayerCount;currentLayer++)
    {
        uint32_t neuronsInThisLayer=currentLayer==candidateGateTotalLayerCount-1?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[currentLayer];
        uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[currentLayer-1];
        size_t thisLayerNeuronCountBasedDoubleArraySize=neuronsInThisLayer*sizeof(double);
        size_t thisLayerNeuronCountBasedDoublePointerArraySize=neuronsInThisLayer*sizeof(double*);
        size_t previousLayerNeuronCountBasedDoubleArraySize=neuronsInPreviousLayer*sizeof(double);
        previousCandidateGateBiasWeightDeltas[currentLayer]=(double*)malloc(thisLayerNeuronCountBasedDoubleArraySize);
        previousCandidateGateWeightDeltas[currentLayer]=(double**)malloc(thisLayerNeuronCountBasedDoublePointerArraySize);

        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
        {
            previousCandidateGateBiasWeightDeltas[currentLayer][neuronInThisLayer]=0.0;
            previousCandidateGateWeightDeltas[currentLayer][neuronInThisLayer]=(double*)malloc(previousLayerNeuronCountBasedDoubleArraySize);
            for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                previousCandidateGateWeightDeltas[currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
        }
    }
}

LSTM::~LSTM()
{
    for(uint32_t layer=stateArrayPos-backpropagationSteps;layer<=stateArrayPos;layer++)
        delete states[layer];
    free(states);

    uint32_t inputAndOutputCount=inputCount+outputCount;
    // Forget gate
    for(uint32_t currentLayer=0;currentLayer<forgetGateTotalLayerCount;currentLayer++)
    {
        uint32_t neuronsInThisLayer=currentLayer==forgetGateTotalLayerCount-1?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[currentLayer];
        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
            free(previousForgetGateWeightDeltas[currentLayer][neuronInThisLayer]);
        free(previousForgetGateWeightDeltas[currentLayer]);
    }

    // Input gate
    for(uint32_t currentLayer=0;currentLayer<inputGateTotalLayerCount;currentLayer++)
    {
        uint32_t neuronsInThisLayer=currentLayer==inputGateTotalLayerCount-1?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[currentLayer];
        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
            free(previousInputGateWeightDeltas[currentLayer][neuronInThisLayer]);
        free(previousInputGateWeightDeltas[currentLayer]);
    }

    // Output gate
    for(uint32_t currentLayer=0;currentLayer<outputGateTotalLayerCount;currentLayer++)
    {
        uint32_t neuronsInThisLayer=currentLayer==outputGateTotalLayerCount-1?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[currentLayer];
        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
            free(previousOutputGateWeightDeltas[currentLayer][neuronInThisLayer]);
        free(previousOutputGateWeightDeltas[currentLayer]);
    }

    // Candidate gate
    for(uint32_t currentLayer=0;currentLayer<candidateGateTotalLayerCount;currentLayer++)
    {
        uint32_t neuronsInThisLayer=currentLayer==candidateGateTotalLayerCount-1?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[currentLayer];
        for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
            free(previousCandidateGateWeightDeltas[currentLayer][neuronInThisLayer]);
        free(previousCandidateGateWeightDeltas[currentLayer]);
        free(previousCandidateGateBiasWeightDeltas[currentLayer]);
    }

    free(previousInputGateBiasWeightDeltas);
    free(previousForgetGateBiasWeightDeltas);
    free(previousOutputGateBiasWeightDeltas);
    free(previousCandidateGateBiasWeightDeltas);
    free(previousInputGateWeightDeltas);
    free(previousForgetGateWeightDeltas);
    free(previousOutputGateWeightDeltas);
    free(previousCandidateGateWeightDeltas);

    free(previousInputGateValueSumBiasWeightDeltas);
    free(previousForgetGateValueSumBiasWeightDeltas);
    free(previousOutputGateValueSumBiasWeightDeltas);
    free(previousCandidateGateValueSumBiasWeightDeltas);

    free(forgetGateHiddenLayerNeuronCounts);
    free(inputGateHiddenLayerNeuronCounts);
    free(outputGateHiddenLayerNeuronCounts);
    free(candidateGateHiddenLayerNeuronCounts);
}

double *LSTM::process(double *input)
{
    LSTMState *l=pushState();
    memcpy(l->input,input,inputCount*sizeof(double)); // Store for backpropagation
    bool hasPreviousState=hasState(1);
    LSTMState *previousState=hasPreviousState?getState(1):0;
    double *output=(double*)malloc(outputCount*sizeof(double));
    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        // Calculate gate pre-values
        l->calculateGatePreValues(hasPreviousState?previousState->output:0);

        // Calculate forget gate value

        double forgetGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            forgetGateValueSum+=l->forgetGatePreValues[cell][i]; // Single-layer version: forgetGateValueSum+=l->forgetGateWeights[cell][i]*input[i];
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                forgetGateValueSum+=l->forgetGatePreValues[cell][inputCount+i]; // Single-layer version: forgetGateValueSum+=l->forgetGateWeights[cell][inputCount+i]*previousState->output[i]
        }
        l->forgetGateValues[cell]=sig(forgetGateValueSum+l->forgetGateValueSumBiasWeights[cell]);

        // Calculate input gate value

        double inputGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            inputGateValueSum+=l->inputGatePreValues[cell][i]; // Single-layer version: inputGateValueSum+=l->inputGateWeights[cell][i]*input[i]
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                inputGateValueSum+=l->inputGatePreValues[cell][inputCount+i]; // Single-layer version: inputGateValueSum+=l->inputGateWeights[cell][inputCount+i]*previousState->output[i]
        }
        l->inputGateValues[cell]=sig(inputGateValueSum+l->inputGateValueSumBiasWeights[cell]);

        // Calculate output gate value

        double outputGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            outputGateValueSum+=l->outputGatePreValues[cell][i]; // Single-layer version: l->outputGateWeights[cell][i]*input[i]
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                outputGateValueSum+=l->outputGatePreValues[cell][inputCount+i]; // Single-layer version: outputGateValueSum+=l->outputGateWeights[cell][inputCount+i]*previousState->output[i]
        }
        l->outputGateValues[cell]=sig(outputGateValueSum+l->outputGateValueSumBiasWeights[cell]);

        // Calculate candidate assessment gate value

        double candidateGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            candidateGateValueSum+=l->candidateGatePreValues[cell][i]; // Single-layer version: l->candidateGateWeights[cell][i]*input[i]
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                candidateGateValueSum+=l->candidateGatePreValues[cell][inputCount+i]; // Single-layer version: l->candidateGateWeights[cell][inputCount+i]*previousState->output[i]
        }
        l->candidateGateValues[cell]=tanh(candidateGateValueSum+l->candidateGateValueSumBiasWeights[cell]);

        // Calculate new cell state

        l->cellStates[cell]=(hasPreviousState?l->forgetGateValues[cell]*previousState->cellStates[cell]/*Old cell state*/:0.0)+l->inputGateValues[cell]*l->candidateGateValues[cell]; // Store for backpropagation

        // Calculate new output value

        // colah's version has a tanh function around the cell state: output[cell]=l->outputGateValues[cell]*tanh(l->cellStates[cell]);
        // Maybe add the tanh?
        output[cell]=l->outputGateValues[cell]*l->cellStates[cell];
        l->output[cell]=output[cell]; // Store for backpropagation
    }
    return output;
}

void LSTM::learn(double **desiredOutputs)
{
    uint32_t availableStepsBack=getAvailableStepsBack();
    // Note that we sum this over all steps, so we do not need the extra time dimension (double**).

    // Differentials of topmost output layer's weights:

    // Dimensions: cells -> layers -> neurons in topmost output layer -> weights of neurons in layer before topmost output layer to neurons in topmost output layer

    double ****wi_diff=(double****)malloc(outputCount*sizeof(double***));
    double ****wf_diff=(double****)malloc(outputCount*sizeof(double***));
    double ****wo_diff=(double****)malloc(outputCount*sizeof(double***));
    double ****wg_diff=(double****)malloc(outputCount*sizeof(double***));

    // Dimensions: cells -> layers -> neurons in layer

    double ***ibi_diff=(double***)malloc(outputCount*sizeof(double**));
    double ***ibf_diff=(double***)malloc(outputCount*sizeof(double**));
    double ***ibo_diff=(double***)malloc(outputCount*sizeof(double**));
    double ***ibg_diff=(double***)malloc(outputCount*sizeof(double**));


    // Error terms

    // Dimensions: cells -> layers -> neurons

    double ***i_errorTerms=(double***)malloc(outputCount*sizeof(double**));
    double ***f_errorTerms=(double***)malloc(outputCount*sizeof(double**));
    double ***o_errorTerms=(double***)malloc(outputCount*sizeof(double**));
    double ***g_errorTerms=(double***)malloc(outputCount*sizeof(double**));

    double *bi_diff=(double*)malloc(outputCount*sizeof(double));
    double *bf_diff=(double*)malloc(outputCount*sizeof(double));
    double *bo_diff=(double*)malloc(outputCount*sizeof(double));
    double *bg_diff=(double*)malloc(outputCount*sizeof(double));
    bool weightsAllocated=false;
    uint32_t inputAndOutputCount=inputCount+outputCount;

    LSTMState *latestState=getCurrentState();

    // This will cycle totalStepCount times, but we need to go backwards, so we use "stepsBack" in combination with "getState(stepsBack)".

    for(uint32_t stepsBack=0;stepsBack<=availableStepsBack;stepsBack++)
    {
        // 0 = current state
        LSTMState *thisState=getState(stepsBack);
        bool hasDeeperState=stepsBack<availableStepsBack;
        bool hasHigherState=stepsBack>0;
        LSTMState *deeperState=hasDeeperState?getState(stepsBack+1):0;
        LSTMState *higherState=hasHigherState?getState(stepsBack-1):0;
        double *_ds=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the cell states
        double *_do=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the output gate values
        double *_di=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the input gate values
        double *_dg=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the candidate gate values
        double *_df=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the forget gate values
        double *_di_input=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the values inside the activation function calls of the input gates (e.g. tanh(x) <- x)
        double *_df_input=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the values inside the activation function calls of the forget gates (e.g. tanh(x) <- x)
        double *_do_input=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the values inside the activation function calls of the output gates (e.g. tanh(x) <- x)
        double *_dg_input=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the values inside the activation function calls of the candidate gates (e.g. tanh(x) <- x)
        // top_diff_is: diff_h = s->bottom_diff_h
        // top_diff_is: diff_s = higherState->bottom_diff_s (topmost: 0)

        // dxc: transpose operation: array of cells=>weights becomes an array of weights=>cells:
        // e.g.:
        // cell1: [weight1_1,weight1_2,weight1_3]
        // cell2: [weight2_1,weight2_2,weight2_3]
        // becomes:
        // weight1: [cell1,cell2]
        // weight2: [cell1,cell2]
        // weight3: [cell1,cell2]
        //
        // Here, we have np.dot(self.param.wi.T, di_input)
        // That means:
        // dxc represents all weights.
        // Each weight i in dxc has as its value: sum over cells*(sum of the four weights of a cell that have the index i (i,f,o,g), each multiplied by their cell's and weight group's (i,f,o, or g) respective derivative w.r.t. the input)).


        // What we need to do is to calculate the derivative of the loss function w.r.t. the biases of the gates,
        // and the weights and biases of the four feedforward neural networks

        double *dxc=(double*)malloc((inputAndOutputCount)*sizeof(double)); // Derivative of loss function with respect to each single input/previous output value
        bool dxcWeightsSet=false;

        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            // For each cell:
            double diff_s=hasHigherState?higherState->bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastCellStates[cell]:0.0;
            double diff_h=2.0*(thisState->output[cell]-desiredOutputs[availableStepsBack-stepsBack][cell]);
            if(hasHigherState)
                diff_h+=higherState->bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastOutputs[cell];

            _ds[cell]=thisState->outputGateValues[cell]*diff_h+diff_s;
            _do[cell]=thisState->cellStates[cell]*diff_h;
            _di[cell]=thisState->candidateGateValues[cell]*_ds[cell];
            _dg[cell]=thisState->inputGateValues[cell]*_ds[cell];
            _df[cell]=(hasDeeperState?deeperState->cellStates[cell]:0.0)*_ds[cell];
            _di_input[cell]=(1.0-thisState->inputGateValues[cell])*thisState->inputGateValues[cell]*_di[cell];
            _df_input[cell]=(1.0-thisState->forgetGateValues[cell])*thisState->forgetGateValues[cell]*_df[cell];
            _do_input[cell]=(1.0-thisState->outputGateValues[cell])*thisState->outputGateValues[cell]*_do[cell];
            _dg_input[cell]=(1.0-pow(thisState->candidateGateValues[cell],2.0))*_dg[cell];

            if(!weightsAllocated)
            {
                bi_diff[cell]=0.0;
                bf_diff[cell]=0.0;
                bo_diff[cell]=0.0;
                bg_diff[cell]=0.0;
            }

            bi_diff[cell]+=_di_input[cell];
            bf_diff[cell]+=_df_input[cell];
            bo_diff[cell]+=_do_input[cell];
            bg_diff[cell]+=_dg_input[cell];

            if(!weightsAllocated)
            {
                wi_diff[cell]=(double***)malloc(inputGateTotalLayerCount*sizeof(double**));
                wf_diff[cell]=(double***)malloc(forgetGateTotalLayerCount*sizeof(double**));
                wo_diff[cell]=(double***)malloc(outputGateTotalLayerCount*sizeof(double**));
                wg_diff[cell]=(double***)malloc(candidateGateTotalLayerCount*sizeof(double**));

                ibi_diff[cell]=(double**)malloc(inputGateTotalLayerCount*sizeof(double*));
                ibf_diff[cell]=(double**)malloc(forgetGateTotalLayerCount*sizeof(double*));
                ibo_diff[cell]=(double**)malloc(outputGateTotalLayerCount*sizeof(double*));
                ibg_diff[cell]=(double**)malloc(candidateGateTotalLayerCount*sizeof(double*));

                i_errorTerms[cell]=(double**)malloc(inputGateTotalLayerCount*sizeof(double*));
                f_errorTerms[cell]=(double**)malloc(forgetGateTotalLayerCount*sizeof(double*));
                o_errorTerms[cell]=(double**)malloc(outputGateTotalLayerCount*sizeof(double*));
                g_errorTerms[cell]=(double**)malloc(candidateGateTotalLayerCount*sizeof(double*));
            }

            // Forget gate
            for(uint32_t _currentLayer=forgetGateTotalLayerCount;_currentLayer>0;_currentLayer--) // Actual layer number: _currentLayer-1 (_currentLayer must be >=0 during the comparison)
            {
                uint32_t currentLayer=_currentLayer-1;
                uint32_t neuronsInThisLayer=currentLayer==forgetGateTotalLayerCount-1?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[currentLayer];
                uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[currentLayer-1];
                uint32_t neuronsInHigherLayer=currentLayer==forgetGateTotalLayerCount-1?0:((forgetGateTotalLayerCount>=2&&currentLayer==forgetGateTotalLayerCount-2)?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[currentLayer+1]);

                if(!weightsAllocated)
                {
                    wf_diff[cell][currentLayer]=(double**)malloc(neuronsInThisLayer*sizeof(double*));
                    ibf_diff[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    f_errorTerms[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                }

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    if(!weightsAllocated)
                    {
                        wf_diff[cell][currentLayer][neuronInThisLayer]=(double*)malloc(neuronsInPreviousLayer*sizeof(double));
                        ibf_diff[cell][currentLayer][neuronInThisLayer]=0.0;
                    }

                    if(currentLayer==forgetGateTotalLayerCount-1)
                        f_errorTerms[cell][currentLayer][neuronInThisLayer]=_df_input[cell];
                    else
                    {
                        double f_errorTermSum=0.0;

                        // Sum error terms of layer above multiplied by the respective weights

                        for(uint32_t neuronInHigherLayer=0;neuronInHigherLayer<neuronsInHigherLayer;neuronInHigherLayer++)
                            f_errorTermSum+=f_errorTerms[cell][currentLayer+1][neuronInHigherLayer]*thisState->forgetGateLayerWeights[cell][currentLayer][neuronInThisLayer][neuronInHigherLayer]/*Weight of this neuron to the neuron in the higher layer*/;

                        f_errorTerms[cell][currentLayer][neuronInThisLayer]=(1.0-pow(thisState->forgetGateLayerNeuronValues[cell][currentLayer][neuronInThisLayer],2))*f_errorTermSum;
                    }

                    ibf_diff[cell][currentLayer][neuronInThisLayer]+=f_errorTerms[cell][currentLayer][neuronInThisLayer];

                    for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                    {
                        if(!weightsAllocated)
                            wf_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
                        wf_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=f_errorTerms[cell][currentLayer][neuronInThisLayer]*(currentLayer==0?(neuronInPreviousLayer<inputCount?thisState->input[neuronInPreviousLayer]:(hasDeeperState?deeperState->output[neuronInPreviousLayer-inputCount]:0.0)):thisState->forgetGateLayerNeuronValues[cell][currentLayer-1][neuronInPreviousLayer]);
                    }
                }
            }

            // Input gate
            for(uint32_t _currentLayer=inputGateTotalLayerCount;_currentLayer>0;_currentLayer--) // Actual layer number: _currentLayer-1 (_currentLayer must be >=0 during the comparison)
            {
                uint32_t currentLayer=_currentLayer-1;
                uint32_t neuronsInThisLayer=currentLayer==inputGateTotalLayerCount-1?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[currentLayer];
                uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[currentLayer-1];
                uint32_t neuronsInHigherLayer=currentLayer==inputGateTotalLayerCount-1?0:((inputGateTotalLayerCount>=2&&currentLayer==inputGateTotalLayerCount-2)?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[currentLayer+1]);

                if(!weightsAllocated)
                {
                    wi_diff[cell][currentLayer]=(double**)malloc(neuronsInThisLayer*sizeof(double*));
                    ibi_diff[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    i_errorTerms[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                }

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    if(!weightsAllocated)
                    {
                        wi_diff[cell][currentLayer][neuronInThisLayer]=(double*)malloc(neuronsInPreviousLayer*sizeof(double));
                        ibi_diff[cell][currentLayer][neuronInThisLayer]=0.0;
                    }

                    if(currentLayer==inputGateTotalLayerCount-1)
                        i_errorTerms[cell][currentLayer][neuronInThisLayer]=_di_input[cell];
                    else
                    {
                        double i_errorTermSum=0.0;

                        // Sum error terms of layer above multiplied by the respective weights

                        for(uint32_t neuronInHigherLayer=0;neuronInHigherLayer<neuronsInHigherLayer;neuronInHigherLayer++)
                            i_errorTermSum+=i_errorTerms[cell][currentLayer+1][neuronInHigherLayer]*thisState->inputGateLayerWeights[cell][currentLayer][neuronInThisLayer][neuronInHigherLayer]/*Weight of this neuron to the neuron in the higher layer*/;

                        i_errorTerms[cell][currentLayer][neuronInThisLayer]=(1.0-pow(thisState->inputGateLayerNeuronValues[cell][currentLayer][neuronInThisLayer],2))*i_errorTermSum;
                    }

                    ibi_diff[cell][currentLayer][neuronInThisLayer]+=i_errorTerms[cell][currentLayer][neuronInThisLayer];

                    for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                    {
                        if(!weightsAllocated)
                            wi_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
                        wi_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=i_errorTerms[cell][currentLayer][neuronInThisLayer]*(currentLayer==0?(neuronInPreviousLayer<inputCount?thisState->input[neuronInPreviousLayer]:(hasDeeperState?deeperState->output[neuronInPreviousLayer-inputCount]:0.0)):thisState->inputGateLayerNeuronValues[cell][currentLayer-1][neuronInPreviousLayer]);
                    }
                }
            }

            // Output gate
            for(uint32_t _currentLayer=outputGateTotalLayerCount;_currentLayer>0;_currentLayer--) // Actual layer number: _currentLayer-1 (_currentLayer must be >=0 during the comparison)
            {
                uint32_t currentLayer=_currentLayer-1;
                uint32_t neuronsInThisLayer=currentLayer==outputGateTotalLayerCount-1?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[currentLayer];
                uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[currentLayer-1];
                uint32_t neuronsInHigherLayer=currentLayer==outputGateTotalLayerCount-1?0:((outputGateTotalLayerCount>=2&&currentLayer==outputGateTotalLayerCount-2)?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[currentLayer+1]);

                if(!weightsAllocated)
                {
                    wo_diff[cell][currentLayer]=(double**)malloc(neuronsInThisLayer*sizeof(double*));
                    ibo_diff[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    o_errorTerms[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                }

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    if(!weightsAllocated)
                    {
                        wo_diff[cell][currentLayer][neuronInThisLayer]=(double*)malloc(neuronsInPreviousLayer*sizeof(double));
                        ibo_diff[cell][currentLayer][neuronInThisLayer]=0.0;
                    }

                    if(currentLayer==outputGateTotalLayerCount-1)
                        o_errorTerms[cell][currentLayer][neuronInThisLayer]=_do_input[cell];
                    else
                    {
                        double o_errorTermSum=0.0;

                        // Sum error terms of layer above multiplied by the respective weights

                        for(uint32_t neuronInHigherLayer=0;neuronInHigherLayer<neuronsInHigherLayer;neuronInHigherLayer++)
                            o_errorTermSum+=o_errorTerms[cell][currentLayer+1][neuronInHigherLayer]*thisState->outputGateLayerWeights[cell][currentLayer][neuronInThisLayer][neuronInHigherLayer]/*Weight of this neuron to the neuron in the higher layer*/;

                        o_errorTerms[cell][currentLayer][neuronInThisLayer]=(1.0-pow(thisState->outputGateLayerNeuronValues[cell][currentLayer][neuronInThisLayer],2))*o_errorTermSum;
                    }

                    ibo_diff[cell][currentLayer][neuronInThisLayer]+=o_errorTerms[cell][currentLayer][neuronInThisLayer];

                    for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                    {
                        if(!weightsAllocated)
                            wo_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
                        wo_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=o_errorTerms[cell][currentLayer][neuronInThisLayer]*(currentLayer==0?(neuronInPreviousLayer<inputCount?thisState->input[neuronInPreviousLayer]:(hasDeeperState?deeperState->output[neuronInPreviousLayer-inputCount]:0.0)):thisState->outputGateLayerNeuronValues[cell][currentLayer-1][neuronInPreviousLayer]);
                    }
                }
            }

            // Candidate gate
            for(uint32_t _currentLayer=candidateGateTotalLayerCount;_currentLayer>0;_currentLayer--) // Actual layer number: _currentLayer-1 (_currentLayer must be >=0 during the comparison)
            {
                uint32_t currentLayer=_currentLayer-1;
                uint32_t neuronsInThisLayer=currentLayer==candidateGateTotalLayerCount-1?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[currentLayer];
                uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[currentLayer-1];
                uint32_t neuronsInHigherLayer=currentLayer==candidateGateTotalLayerCount-1?0:((candidateGateTotalLayerCount>=2&&currentLayer==candidateGateTotalLayerCount-2)?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[currentLayer+1]);

                if(!weightsAllocated)
                {
                    wg_diff[cell][currentLayer]=(double**)malloc(neuronsInThisLayer*sizeof(double*));
                    ibg_diff[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    g_errorTerms[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                }

                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    if(!weightsAllocated)
                    {
                        wg_diff[cell][currentLayer][neuronInThisLayer]=(double*)malloc(neuronsInPreviousLayer*sizeof(double));
                        ibg_diff[cell][currentLayer][neuronInThisLayer]=0.0;
                    }

                    if(currentLayer==candidateGateTotalLayerCount-1)
                        g_errorTerms[cell][currentLayer][neuronInThisLayer]=_dg_input[cell];
                    else
                    {
                        double g_errorTermSum=0.0;

                        // Sum error terms of layer above multiplied by the respective weights

                        for(uint32_t neuronInHigherLayer=0;neuronInHigherLayer<neuronsInHigherLayer;neuronInHigherLayer++)
                            g_errorTermSum+=g_errorTerms[cell][currentLayer+1][neuronInHigherLayer]*thisState->candidateGateLayerWeights[cell][currentLayer][neuronInThisLayer][neuronInHigherLayer]/*Weight of this neuron to the neuron in the higher layer*/;

                        g_errorTerms[cell][currentLayer][neuronInThisLayer]=(1.0-pow(thisState->candidateGateLayerNeuronValues[cell][currentLayer][neuronInThisLayer],2))*g_errorTermSum;
                    }

                    ibg_diff[cell][currentLayer][neuronInThisLayer]+=g_errorTerms[cell][currentLayer][neuronInThisLayer];

                    for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                    {
                        if(!weightsAllocated)
                            wg_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
                        wg_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=g_errorTerms[cell][currentLayer][neuronInThisLayer]*(currentLayer==0?(neuronInPreviousLayer<inputCount?thisState->input[neuronInPreviousLayer]:(hasDeeperState?deeperState->output[neuronInPreviousLayer-inputCount]:0.0)):thisState->candidateGateLayerNeuronValues[cell][currentLayer-1][neuronInPreviousLayer]);
                    }
                }
            }

            // Calculate derivatives of loss function w.r.t. the inputs received from the last state

            // The bottommost layer has the inputs/outputs of the cell as its inputs.
            // The bottommost layer's weights are used to feed in the inputs into the bottommost layer of the neural network (by multiplying them by the bottommost layer's weights).
            // => Calculate error term of bottommost layer

            double i_errorTermSum=0.0;
            double f_errorTermSum=0.0;
            double o_errorTermSum=0.0;
            double g_errorTermSum=0.0;

            // Sum error terms of layer above multiplied by the respective weights

            uint32_t neuronsInForgetGateBottommostLayer=forgetGateHiddenLayerCount==0?inputAndOutputCount:forgetGateHiddenLayerNeuronCounts[0];
            uint32_t neuronsInInputGateBottommostLayer=inputGateHiddenLayerCount==0?inputAndOutputCount:inputGateHiddenLayerNeuronCounts[0];
            uint32_t neuronsInOutputGateBottommostLayer=outputGateHiddenLayerCount==0?inputAndOutputCount:outputGateHiddenLayerNeuronCounts[0];
            uint32_t neuronsInCandidateGateBottommostLayer=candidateGateHiddenLayerCount==0?inputAndOutputCount:candidateGateHiddenLayerNeuronCounts[0];

            // Forget gate
            for(uint32_t neuronInBottommostLayer=0;neuronInBottommostLayer<neuronsInForgetGateBottommostLayer;neuronInBottommostLayer++)
            {
                for(uint32_t weightInputOrOutput=0;weightInputOrOutput<inputAndOutputCount;weightInputOrOutput++)
                    f_errorTermSum+=f_errorTerms[cell][0 /*Bottommost layer*/][neuronInBottommostLayer]*thisState->forgetGateLayerWeights[cell][0 /*Bottommost layer*/][neuronInBottommostLayer][weightInputOrOutput]; // Weight of this neuron to the neuron in the higher layer
            }

            // Input gate
            for(uint32_t neuronInBottommostLayer=0;neuronInBottommostLayer<neuronsInInputGateBottommostLayer;neuronInBottommostLayer++)
            {
                for(uint32_t weightInputOrOutput=0;weightInputOrOutput<inputAndOutputCount;weightInputOrOutput++)
                    i_errorTermSum+=i_errorTerms[cell][0 /*Bottommost layer*/][neuronInBottommostLayer]*thisState->inputGateLayerWeights[cell][0 /*Bottommost layer*/][neuronInBottommostLayer][weightInputOrOutput]; // Weight of this neuron to the neuron in the higher layer
            }

            // Output gate
            for(uint32_t neuronInBottommostLayer=0;neuronInBottommostLayer<neuronsInOutputGateBottommostLayer;neuronInBottommostLayer++)
            {
                for(uint32_t weightInputOrOutput=0;weightInputOrOutput<inputAndOutputCount;weightInputOrOutput++)
                    o_errorTermSum+=o_errorTerms[cell][0 /*Bottommost layer*/][neuronInBottommostLayer]*thisState->outputGateLayerWeights[cell][0 /*Bottommost layer*/][neuronInBottommostLayer][weightInputOrOutput]; // Weight of this neuron to the neuron in the higher layer
            }

            // Output gate
            for(uint32_t neuronInBottommostLayer=0;neuronInBottommostLayer<neuronsInCandidateGateBottommostLayer;neuronInBottommostLayer++)
            {
                for(uint32_t weightInputOrOutput=0;weightInputOrOutput<inputAndOutputCount;weightInputOrOutput++)
                    g_errorTermSum+=g_errorTerms[cell][0 /*Bottommost layer*/][neuronInBottommostLayer]*thisState->candidateGateLayerWeights[cell][0 /*Bottommost layer*/][neuronInBottommostLayer][weightInputOrOutput]; // Weight of this neuron to the neuron in the higher layer
            }

            for(uint32_t weightInput=0;weightInput<inputCount;weightInput++)
            {
                if(!dxcWeightsSet)
                    dxc[weightInput]=0.0;
                dxc[weightInput]=i_errorTermSum+f_errorTermSum+o_errorTermSum+g_errorTermSum;
            }
            for(uint32_t weightOutput=0;weightOutput<outputCount;weightOutput++)
            {
                if(!dxcWeightsSet)
                    dxc[inputCount+weightOutput]=0.0;
                dxc[inputCount+weightOutput]=i_errorTermSum+f_errorTermSum+o_errorTermSum+g_errorTermSum;
            }

            // bottom_diff_s:
            thisState->bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastCellStates[cell]=_ds[cell]*thisState->forgetGateValues[cell];
            if(!dxcWeightsSet)
                dxcWeightsSet=true;
        }

        // bottom_diff_x:
        memcpy(thisState->bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToInputs,dxc,inputCount*sizeof(double));
        // bottom_diff_h:
        memcpy(thisState->bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastOutputs,dxc+inputCount,outputCount*sizeof(double));

        free(dxc);
        free(_ds);
        free(_do);
        free(_di);
        free(_dg);
        free(_df);
        free(_di_input);
        free(_df_input);
        free(_do_input);
        free(_dg_input);
        if(!weightsAllocated)
            weightsAllocated=true;
    }

    double ***gateLayerWeights;
    double **gateLayerBiasWeights;
    double ***previousGateWeightDeltas;
    double **previousGateBiasWeightDeltas;
    double ***gateLayerWeightDiffs;
    double **gateLayerBiasWeightDiffs;
    double ***gateErrorTerms;
    uint32_t gateHiddenLayerCount;
    uint32_t *gateHiddenLayerNeuronCounts;
    double gateNetworkLearningRate;
    double gateNetworkMomentum;
    double gateNetworkWeightDecay;

    // Now that we have cycled through all states, apply all changes:

    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        // For each gate
        for(uint8_t gate=1;gate<=4;gate++)
        {
            if(gate==1)
            {
                // Forget gate
                gateLayerWeights=latestState->forgetGateLayerWeights[cell];
                gateLayerBiasWeights=latestState->forgetGateLayerBiasWeights[cell];
                previousGateWeightDeltas=previousForgetGateWeightDeltas;
                previousGateBiasWeightDeltas=previousForgetGateBiasWeightDeltas;
                gateLayerWeightDiffs=wf_diff[cell];
                gateLayerBiasWeightDiffs=ibf_diff[cell];
                gateHiddenLayerCount=forgetGateHiddenLayerCount;
                gateHiddenLayerNeuronCounts=forgetGateHiddenLayerNeuronCounts;
                gateErrorTerms=f_errorTerms;
                gateNetworkLearningRate=forgetGateNetworkLearningRate;
                gateNetworkMomentum=forgetGateNetworkMomentum;
                gateNetworkWeightDecay=forgetGateNetworkWeightDecay;
            }
            else if(gate==2)
            {
                // Input gate

                gateLayerWeights=latestState->inputGateLayerWeights[cell];
                gateLayerBiasWeights=latestState->inputGateLayerBiasWeights[cell];
                previousGateWeightDeltas=previousInputGateWeightDeltas;
                previousGateBiasWeightDeltas=previousInputGateBiasWeightDeltas;
                gateLayerWeightDiffs=wi_diff[cell];
                gateLayerBiasWeightDiffs=ibi_diff[cell];
                gateHiddenLayerCount=inputGateHiddenLayerCount;
                gateHiddenLayerNeuronCounts=inputGateHiddenLayerNeuronCounts;
                gateErrorTerms=i_errorTerms;
                gateNetworkLearningRate=inputGateNetworkLearningRate;
                gateNetworkMomentum=inputGateNetworkMomentum;
                gateNetworkWeightDecay=inputGateNetworkWeightDecay;
            }
            else if(gate==3)
            {
                // Output gate

                gateLayerWeights=latestState->outputGateLayerWeights[cell];
                gateLayerBiasWeights=latestState->outputGateLayerBiasWeights[cell];
                previousGateWeightDeltas=previousOutputGateWeightDeltas;
                previousGateBiasWeightDeltas=previousOutputGateBiasWeightDeltas;
                gateLayerWeightDiffs=wo_diff[cell];
                gateLayerBiasWeightDiffs=ibo_diff[cell];
                gateHiddenLayerCount=outputGateHiddenLayerCount;
                gateHiddenLayerNeuronCounts=outputGateHiddenLayerNeuronCounts;
                gateErrorTerms=o_errorTerms;
                gateNetworkLearningRate=outputGateNetworkLearningRate;
                gateNetworkMomentum=outputGateNetworkMomentum;
                gateNetworkWeightDecay=outputGateNetworkWeightDecay;
            }
            else // if(gate==4)
            {
                // Candidate gate

                gateLayerWeights=latestState->candidateGateLayerWeights[cell];
                gateLayerBiasWeights=latestState->candidateGateLayerBiasWeights[cell];
                previousGateWeightDeltas=previousCandidateGateWeightDeltas;
                previousGateBiasWeightDeltas=previousCandidateGateBiasWeightDeltas;
                gateLayerWeightDiffs=wg_diff[cell];
                gateLayerBiasWeightDiffs=ibg_diff[cell];
                gateHiddenLayerCount=candidateGateHiddenLayerCount;
                gateHiddenLayerNeuronCounts=candidateGateHiddenLayerNeuronCounts;
                gateErrorTerms=g_errorTerms;
                gateNetworkLearningRate=candidateGateNetworkLearningRate;
                gateNetworkMomentum=candidateGateNetworkMomentum;
                gateNetworkWeightDecay=candidateGateNetworkWeightDecay;
            }

            for(uint32_t _currentLayer=gateHiddenLayerCount+1/*Include topmost output layer*/;_currentLayer>0;_currentLayer--)
            {
                uint32_t currentLayer=_currentLayer-1;
                uint32_t neuronsInThisLayer=currentLayer==gateHiddenLayerCount/*Is topmost output layer?*/?inputAndOutputCount:gateHiddenLayerNeuronCounts[currentLayer];
                uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:gateHiddenLayerNeuronCounts[currentLayer-1];
                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Adjust bias of this neuron
                    double currentBiasWeight=gateLayerBiasWeights[currentLayer][neuronInThisLayer];
                    double previousBiasWeightDelta=previousGateBiasWeightDeltas[currentLayer][neuronInThisLayer];
                    double biasWeightDelta=(1.0-gateNetworkMomentum)*-gateNetworkLearningRate*gateLayerBiasWeightDiffs[currentLayer][neuronInThisLayer]+gateNetworkMomentum*previousBiasWeightDelta-gateNetworkWeightDecay*currentBiasWeight;
                    gateLayerBiasWeights[currentLayer][neuronInThisLayer]+=biasWeightDelta;
                    previousGateBiasWeightDeltas[currentLayer][neuronInThisLayer]=biasWeightDelta;
                    for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                    {
                        // Adjust weight from neuronInPreviousLayer to neuronInThisLayer
                        double currentWeight=gateLayerWeights[currentLayer][neuronInThisLayer][neuronInPreviousLayer];
                        double previousWeightDelta=previousGateWeightDeltas[currentLayer][neuronInThisLayer][neuronInPreviousLayer];
                        double weightDelta=(1.0-gateNetworkMomentum)*-gateNetworkLearningRate*gateLayerWeightDiffs[currentLayer][neuronInThisLayer][neuronInPreviousLayer]+gateNetworkMomentum*previousWeightDelta-gateNetworkWeightDecay*currentWeight;
                        gateLayerWeights[currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=weightDelta;
                        previousGateWeightDeltas[currentLayer][neuronInThisLayer][neuronInPreviousLayer]=weightDelta;
                    }
                    free(gateLayerWeightDiffs[currentLayer][neuronInThisLayer]);
                }
                free(gateLayerBiasWeightDiffs[currentLayer]);
                free(gateLayerWeightDiffs[currentLayer]);
                free(gateErrorTerms[cell][currentLayer]);
            }
            free(gateLayerBiasWeightDiffs);
            free(gateLayerWeightDiffs);
        }

        // Free error terms
        free(i_errorTerms[cell]);
        free(f_errorTerms[cell]);
        free(o_errorTerms[cell]);
        free(g_errorTerms[cell]);

        double previousInputGateValueSumBiasWeightDelta=previousInputGateValueSumBiasWeightDeltas[cell];
        double previousForgetGateValueSumBiasWeightDelta=previousForgetGateValueSumBiasWeightDeltas[cell];
        double previousOutputGateValueSumBiasWeightDelta=previousOutputGateValueSumBiasWeightDeltas[cell];
        double previousCandidateGateValueSumBiasWeightDelta=previousCandidateGateValueSumBiasWeightDeltas[cell];
        double currentInputGateValueSumBiasWeight=latestState->inputGateValueSumBiasWeights[cell];
        double currentForgetGateValueSumBiasWeight=latestState->forgetGateValueSumBiasWeights[cell];
        double currentOutputGateValueSumBiasWeight=latestState->outputGateValueSumBiasWeights[cell];
        double currentCandidateGateValueSumBiasWeight=latestState->candidateGateValueSumBiasWeights[cell];
        double inputGateValueSumBiasWeightDelta=(1.0-momentum)*-learningRate*bi_diff[cell]+momentum*previousForgetGateValueSumBiasWeightDelta*-weightDecay*currentInputGateValueSumBiasWeight;
        double forgetGateValueSumBiasWeightDelta=(1.0-momentum)*-learningRate*bf_diff[cell]+momentum*previousInputGateValueSumBiasWeightDelta-weightDecay*currentForgetGateValueSumBiasWeight;
        double outputGateValueSumBiasWeightDelta=(1.0-momentum)*-learningRate*bo_diff[cell]+momentum*previousOutputGateValueSumBiasWeightDelta-weightDecay*currentOutputGateValueSumBiasWeight;
        double candidateGateValueSumBiasWeightDelta=(1.0-momentum)*-learningRate*bg_diff[cell]+momentum*previousCandidateGateValueSumBiasWeightDelta-weightDecay*currentCandidateGateValueSumBiasWeight;
        latestState->inputGateValueSumBiasWeights[cell]+=inputGateValueSumBiasWeightDelta;
        latestState->forgetGateValueSumBiasWeights[cell]+=forgetGateValueSumBiasWeightDelta;
        latestState->outputGateValueSumBiasWeights[cell]+=outputGateValueSumBiasWeightDelta;
        latestState->candidateGateValueSumBiasWeights[cell]+=candidateGateValueSumBiasWeightDelta;
        previousInputGateValueSumBiasWeightDeltas[cell]=forgetGateValueSumBiasWeightDelta;
        previousForgetGateValueSumBiasWeightDeltas[cell]=inputGateValueSumBiasWeightDelta;
        previousOutputGateValueSumBiasWeightDeltas[cell]=outputGateValueSumBiasWeightDelta;
        previousCandidateGateValueSumBiasWeightDeltas[cell]=candidateGateValueSumBiasWeightDelta;
    }

    free(i_errorTerms);
    free(f_errorTerms);
    free(o_errorTerms);
    free(g_errorTerms);
    free(wi_diff);
    free(wf_diff);
    free(wo_diff);
    free(wg_diff);
    free(ibi_diff);
    free(ibf_diff);
    free(ibo_diff);
    free(ibg_diff);
    free(bi_diff);
    free(bf_diff);
    free(bo_diff);
    free(bg_diff);
}

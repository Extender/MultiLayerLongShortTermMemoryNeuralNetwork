#include "lstm.h"

double LSTM::sig(double input)
{
    // Derivative: sig(input)*(1.0-sig(input))
    return 1.0/(1.0+pow(M_E,-input));
}

double LSTM::tanh(double input)
{
    // Derivative: 1.0-((tanh(input))^2)
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
    srand(time(NULL));
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
    LSTMState *newState=stateArrayPos>0/*Has previous state?*/?new LSTMState(getState(1)):new LSTMState(0,inputCount,outputCount,hiddenLayerCount,hiddenLayerNeuronCounts);
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

LSTM::LSTM(uint32_t _inputCount, uint32_t _outputCount, uint32_t _backpropagationSteps, double _learningRate, uint32_t _hiddenLayerCount, uint32_t *_hiddenLayerNeuronCounts)
{
    inputCount=_inputCount;
    outputCount=_outputCount;
    uint32_t inputAndOutputCount=inputCount+outputCount;
    backpropagationSteps=_backpropagationSteps;
    learningRate=_learningRate;

    stateArraySize=2*backpropagationSteps+1 /*One for the current state.*/;
    stateArrayPos=0xffffffff;
    states=(LSTMState**)malloc(stateArraySize*sizeof(LSTMState*));

    hiddenLayerCount=_hiddenLayerCount;
    totalLayerCount=hiddenLayerCount+1;

    // Copy hidden layer neuron counts (to avoid errors)
    uint32_t hiddenLayerNeuronCountArraySize=hiddenLayerCount*sizeof(uint32_t);
    hiddenLayerNeuronCounts=(uint32_t*)malloc(hiddenLayerNeuronCountArraySize);
    if(_hiddenLayerNeuronCounts==0)
    {
        for(uint32_t hiddenLayer=0;hiddenLayer<hiddenLayerCount;hiddenLayer++)
            hiddenLayerNeuronCounts[hiddenLayer]=inputAndOutputCount;
    }
    else
        memcpy(hiddenLayerNeuronCounts,_hiddenLayerNeuronCounts,hiddenLayerNeuronCountArraySize);
}

LSTM::~LSTM()
{
    for(uint32_t layer=stateArrayPos-backpropagationSteps;layer<=stateArrayPos;layer++)
        delete states[layer];
    free(states);
    free(hiddenLayerNeuronCounts);
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
        l->forgetGateValues[cell]=sig(forgetGateValueSum+l->forgetGateBiasWeights[cell]);

        // Calculate input gate value

        double inputGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            inputGateValueSum+=l->inputGatePreValues[cell][i]; // Single-layer version: inputGateValueSum+=l->inputGateWeights[cell][i]*input[i]
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                inputGateValueSum+=l->inputGatePreValues[cell][inputCount+i]; // Single-layer version: inputGateValueSum+=l->inputGateWeights[cell][inputCount+i]*previousState->output[i]
        }
        l->inputGateValues[cell]=sig(inputGateValueSum+l->inputGateBiasWeights[cell]);

        // Calculate output gate value

        double outputGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            outputGateValueSum+=l->outputGatePreValues[cell][i]; // Single-layer version: l->outputGateWeights[cell][i]*input[i]
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                outputGateValueSum+=l->outputGatePreValues[cell][inputCount+i]; // Single-layer version: outputGateValueSum+=l->outputGateWeights[cell][inputCount+i]*previousState->output[i]
        }
        l->outputGateValues[cell]=sig(outputGateValueSum+l->outputGateBiasWeights[cell]);

        // Calculate candidate assessment gate value

        double candidateGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            candidateGateValueSum+=l->candidateGatePreValues[cell][i]; // Single-layer version: l->candidateGateWeights[cell][i]*input[i]
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                candidateGateValueSum+=l->candidateGatePreValues[cell][inputCount+i]; // Single-layer version: l->candidateGateWeights[cell][inputCount+i]*previousState->output[i]
        }
        l->candidateGateValues[cell]=tanh(candidateGateValueSum+l->candidateGateBiasWeights[cell]);

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
    // Check for any non-initialized arrays

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
        bool hasDeeperState=stepsBack<availableStepsBack; // Or hasNext? Should we use the next value instead?
        bool hasHigherState=stepsBack>0; // Or hasNext? Should we use the next value instead?
        LSTMState *deeperState=hasDeeperState?getState(stepsBack+1):0;
        LSTMState *higherState=hasHigherState?getState(stepsBack-1):0;
        double *_ds=(double*)malloc(outputCount*sizeof(double));
        double *_do=(double*)malloc(outputCount*sizeof(double));
        double *_di=(double*)malloc(outputCount*sizeof(double));
        double *_dg=(double*)malloc(outputCount*sizeof(double));
        double *_df=(double*)malloc(outputCount*sizeof(double));
        double *_di_input=(double*)malloc(outputCount*sizeof(double));
        double *_df_input=(double*)malloc(outputCount*sizeof(double));
        double *_do_input=(double*)malloc(outputCount*sizeof(double));
        double *_dg_input=(double*)malloc(outputCount*sizeof(double));
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
        // Each weight i in dxc has as its value: sum over cells(sum of the four weights of a cell that have the index i (i,f,o,g), each multiplied by their cell's and weight group's (i,f,o, or g) respective derivative w.r.t. the input)).


        // What we need to do is to calculate the derivative of the loss function w.r.t. the biases of the gates,
        // and the weights and biases of the four feedforward neural networks.

        double *dxc=(double*)malloc((inputAndOutputCount)*sizeof(double)); // Derivative of loss function with respect to each single input/previous output value.
        bool dxcWeightsSet=false;

        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            // For each cell:
            double diff_s=hasHigherState?higherState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToCellStates[cell]:0.0;
            double diff_h=2.0*(thisState->output[cell]-desiredOutputs[availableStepsBack-stepsBack][cell]); // Taken from bottom_diff, may not be correct!
            if(hasHigherState)
                diff_h+=higherState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToOutputs[cell];

            _ds[cell]=thisState->outputGateValues[cell]*diff_h+diff_s; // Derivative of the loss function w.r.t. the cell states
            _do[cell]=thisState->cellStates[cell]*diff_h; // Derivative of the loss function w.r.t. the output gate's value
            _di[cell]=thisState->candidateGateValues[cell]*_ds[cell]; // Derivative of the loss function w.r.t. the input gate's value
            _dg[cell]=thisState->inputGateValues[cell]*_ds[cell]; // Derivative of the loss function w.r.t. the candidate gate's value
            _df[cell]=(hasDeeperState?deeperState->cellStates[cell]:0.0)*_ds[cell]; // Derivative of the loss function w.r.t. the forget gate's value
            _di_input[cell]=(1.0-thisState->inputGateValues[cell])*thisState->inputGateValues[cell]*_di[cell]; // Derivative of the loss function w.r.t. the value inside the sigmoid function of the input gate
            _df_input[cell]=(1.0-thisState->forgetGateValues[cell])*thisState->forgetGateValues[cell]*_df[cell]; // Derivative of the loss function w.r.t. the value inside the sigmoid function of the forget gate
            _do_input[cell]=(1.0-thisState->outputGateValues[cell])*thisState->outputGateValues[cell]*_do[cell]; // Derivative of the loss function w.r.t. the value inside the sigmoid function of the output gate
            _dg_input[cell]=(1.0-pow(thisState->candidateGateValues[cell],2.0))*_dg[cell]; // Derivative of the loss function w.r.t. the value inside the tanh function of the forget gate

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
                wi_diff[cell]=(double***)malloc(totalLayerCount*sizeof(double**));
                wf_diff[cell]=(double***)malloc(totalLayerCount*sizeof(double**));
                wo_diff[cell]=(double***)malloc(totalLayerCount*sizeof(double**));
                wg_diff[cell]=(double***)malloc(totalLayerCount*sizeof(double**));

                ibi_diff[cell]=(double**)malloc(totalLayerCount*sizeof(double*));
                ibf_diff[cell]=(double**)malloc(totalLayerCount*sizeof(double*));
                ibo_diff[cell]=(double**)malloc(totalLayerCount*sizeof(double*));
                ibg_diff[cell]=(double**)malloc(totalLayerCount*sizeof(double*));

                i_errorTerms[cell]=(double**)malloc(totalLayerCount*sizeof(double*));
                f_errorTerms[cell]=(double**)malloc(totalLayerCount*sizeof(double*));
                o_errorTerms[cell]=(double**)malloc(totalLayerCount*sizeof(double*));
                g_errorTerms[cell]=(double**)malloc(totalLayerCount*sizeof(double*));
            }

            for(uint32_t _currentLayer=totalLayerCount;_currentLayer>0;_currentLayer--) // Actual layer number: _currentLayer-1 (_currentLayer must be >=0 during the comparison)
            {
                uint32_t currentLayer=_currentLayer-1;
                uint32_t neuronsInThisLayer=currentLayer==totalLayerCount-1?inputAndOutputCount:hiddenLayerNeuronCounts[currentLayer];
                uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:hiddenLayerNeuronCounts[currentLayer-1];
                uint32_t neuronsInHigherLayer=currentLayer==totalLayerCount-1?0:((totalLayerCount>=2&&currentLayer==totalLayerCount-2)?inputAndOutputCount:hiddenLayerNeuronCounts[currentLayer+1]);

                if(!weightsAllocated)
                {
                    wi_diff[cell][currentLayer]=(double**)malloc(neuronsInThisLayer*sizeof(double*));
                    wf_diff[cell][currentLayer]=(double**)malloc(neuronsInThisLayer*sizeof(double*));
                    wo_diff[cell][currentLayer]=(double**)malloc(neuronsInThisLayer*sizeof(double*));
                    wg_diff[cell][currentLayer]=(double**)malloc(neuronsInThisLayer*sizeof(double*));

                    ibi_diff[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    ibf_diff[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    ibo_diff[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    ibg_diff[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));

                    i_errorTerms[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    f_errorTerms[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    o_errorTerms[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                    g_errorTerms[cell][currentLayer]=(double*)malloc(neuronsInThisLayer*sizeof(double));
                }


                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    if(!weightsAllocated)
                    {
                        wi_diff[cell][currentLayer][neuronInThisLayer]=(double*)malloc(neuronsInPreviousLayer*sizeof(double));
                        wf_diff[cell][currentLayer][neuronInThisLayer]=(double*)malloc(neuronsInPreviousLayer*sizeof(double));
                        wo_diff[cell][currentLayer][neuronInThisLayer]=(double*)malloc(neuronsInPreviousLayer*sizeof(double));
                        wg_diff[cell][currentLayer][neuronInThisLayer]=(double*)malloc(neuronsInPreviousLayer*sizeof(double));

                        ibi_diff[cell][currentLayer][neuronInThisLayer]=0.0;
                        ibf_diff[cell][currentLayer][neuronInThisLayer]=0.0;
                        ibo_diff[cell][currentLayer][neuronInThisLayer]=0.0;
                        ibg_diff[cell][currentLayer][neuronInThisLayer]=0.0;
                    }

                    if(currentLayer==totalLayerCount-1)
                    {
                        i_errorTerms[cell][currentLayer][neuronInThisLayer]=_di_input[cell];
                        f_errorTerms[cell][currentLayer][neuronInThisLayer]=_df_input[cell];
                        o_errorTerms[cell][currentLayer][neuronInThisLayer]=_do_input[cell];
                        g_errorTerms[cell][currentLayer][neuronInThisLayer]=_dg_input[cell];
                    }
                    else
                    {
                        double i_errorTermSum=0.0;
                        double f_errorTermSum=0.0;
                        double o_errorTermSum=0.0;
                        double g_errorTermSum=0.0;

                        // Sum error terms of layer above multiplied by the respective weights

                        for(uint32_t neuronInHigherLayer=0;neuronInHigherLayer<neuronsInHigherLayer;neuronInHigherLayer++)
                        {
                            // CHECK THIS AGAIN:
                            i_errorTermSum+=i_errorTerms[cell][currentLayer+1][neuronInHigherLayer]*thisState->inputGateLayerWeights[cell][currentLayer][neuronInThisLayer][neuronInHigherLayer]/*Weight of this neuron to the neuron in the higher layer (CHECK THIS!)*/;
                            f_errorTermSum+=f_errorTerms[cell][currentLayer+1][neuronInHigherLayer]*thisState->forgetGateLayerWeights[cell][currentLayer][neuronInThisLayer][neuronInHigherLayer]/*Weight of this neuron to the neuron in the higher layer (CHECK THIS!)*/;
                            o_errorTermSum+=o_errorTerms[cell][currentLayer+1][neuronInHigherLayer]*thisState->outputGateLayerWeights[cell][currentLayer][neuronInThisLayer][neuronInHigherLayer]/*Weight of this neuron to the neuron in the higher layer (CHECK THIS!)*/;
                            g_errorTermSum+=g_errorTerms[cell][currentLayer+1][neuronInHigherLayer]*thisState->candidateGateLayerWeights[cell][currentLayer][neuronInThisLayer][neuronInHigherLayer]/*Weight of this neuron to the neuron in the higher layer (CHECK THIS!)*/;
                        }

                        i_errorTerms[cell][currentLayer][neuronInThisLayer]=(1.0-pow(thisState->inputGateLayerNeuronValues[cell][currentLayer][neuronInThisLayer],2))*i_errorTermSum;
                        f_errorTerms[cell][currentLayer][neuronInThisLayer]=(1.0-pow(thisState->forgetGateLayerNeuronValues[cell][currentLayer][neuronInThisLayer],2))*f_errorTermSum;
                        o_errorTerms[cell][currentLayer][neuronInThisLayer]=(1.0-pow(thisState->outputGateLayerNeuronValues[cell][currentLayer][neuronInThisLayer],2))*o_errorTermSum;
                        g_errorTerms[cell][currentLayer][neuronInThisLayer]=(1.0-pow(thisState->candidateGateLayerNeuronValues[cell][currentLayer][neuronInThisLayer],2))*g_errorTermSum;
                    }

                    ibi_diff[cell][currentLayer][neuronInThisLayer]+=i_errorTerms[cell][currentLayer][neuronInThisLayer];
                    ibf_diff[cell][currentLayer][neuronInThisLayer]+=f_errorTerms[cell][currentLayer][neuronInThisLayer];
                    ibo_diff[cell][currentLayer][neuronInThisLayer]+=o_errorTerms[cell][currentLayer][neuronInThisLayer];
                    ibg_diff[cell][currentLayer][neuronInThisLayer]+=g_errorTerms[cell][currentLayer][neuronInThisLayer];

                    for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                    {
                        if(!weightsAllocated)
                        {
                            wi_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
                            wf_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
                            wo_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
                            wg_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]=0.0;
                        }

                        wi_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=i_errorTerms[cell][currentLayer][neuronInThisLayer]*(currentLayer==0?(neuronInPreviousLayer<inputCount?thisState->input[neuronInPreviousLayer]:(hasDeeperState?deeperState->output[neuronInPreviousLayer-inputCount]:0.0)):thisState->inputGateLayerNeuronValues[cell][currentLayer-1][neuronInPreviousLayer]);
                        wf_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=f_errorTerms[cell][currentLayer][neuronInThisLayer]*(currentLayer==0?(neuronInPreviousLayer<inputCount?thisState->input[neuronInPreviousLayer]:(hasDeeperState?deeperState->output[neuronInPreviousLayer-inputCount]:0.0)):thisState->forgetGateLayerNeuronValues[cell][currentLayer-1][neuronInPreviousLayer]);
                        wo_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=o_errorTerms[cell][currentLayer][neuronInThisLayer]*(currentLayer==0?(neuronInPreviousLayer<inputCount?thisState->input[neuronInPreviousLayer]:(hasDeeperState?deeperState->output[neuronInPreviousLayer-inputCount]:0.0)):thisState->outputGateLayerNeuronValues[cell][currentLayer-1][neuronInPreviousLayer]);
                        wg_diff[cell][currentLayer][neuronInThisLayer][neuronInPreviousLayer]+=g_errorTerms[cell][currentLayer][neuronInThisLayer]*(currentLayer==0?(neuronInPreviousLayer<inputCount?thisState->input[neuronInPreviousLayer]:(hasDeeperState?deeperState->output[neuronInPreviousLayer-inputCount]:0.0)):thisState->candidateGateLayerNeuronValues[cell][currentLayer-1][neuronInPreviousLayer]);
                    }
                }
            }

            // Adjust dxc (CHECK THIS)!

            // The bottommost layer has the inputs/outputs of the cell as its inputs.
            // The bottommost layer's weights are used to feed in the inputs into the bottommost layer of the neural network (by multiplying them by the bottommost layer's weights).
            // => Calculate error term of bottommost layer.

            double i_errorTermSum=0.0;
            double f_errorTermSum=0.0;
            double o_errorTermSum=0.0;
            double g_errorTermSum=0.0;

            // Sum error terms of layer above multiplied by the respective weights

            uint32_t neuronsInBottommostLayer=hiddenLayerCount==0?inputAndOutputCount:hiddenLayerNeuronCounts[0];

            for(uint32_t neuronInBottommostLayer=0;neuronInBottommostLayer<neuronsInBottommostLayer;neuronInBottommostLayer++)
            {
                // CHECK THIS AGAIN:
                for(uint32_t weightInputOrOutput=0;weightInputOrOutput<inputAndOutputCount;weightInputOrOutput++)
                {
                    i_errorTermSum+=i_errorTerms[cell][0 /*Bottommost layer*/][neuronInBottommostLayer]*thisState->inputGateLayerWeights[cell][0 /*Bottommost layer*/][neuronInBottommostLayer][weightInputOrOutput] /*Weight of this neuron to the neuron in the higher layer (CHECK THIS!)*/;
                    f_errorTermSum+=f_errorTerms[cell][0 /*Bottommost layer*/][neuronInBottommostLayer]*thisState->forgetGateLayerWeights[cell][0 /*Bottommost layer*/][neuronInBottommostLayer][weightInputOrOutput] /*No weight premuliplication for inputs.*/ /*Weight of this neuron to the neuron in the higher layer (CHECK THIS!)*/;
                    o_errorTermSum+=o_errorTerms[cell][0 /*Bottommost layer*/][neuronInBottommostLayer]*thisState->outputGateLayerWeights[cell][0 /*Bottommost layer*/][neuronInBottommostLayer][weightInputOrOutput] /*No weight premuliplication for inputs.*/ /*Weight of this neuron to the neuron in the higher layer (CHECK THIS!)*/;
                    g_errorTermSum+=g_errorTerms[cell][0 /*Bottommost layer*/][neuronInBottommostLayer]*thisState->candidateGateLayerWeights[cell][0 /*Bottommost layer*/][neuronInBottommostLayer][weightInputOrOutput] /*No weight premuliplication for inputs.*/ /*Weight of this neuron to the neuron in the higher layer (CHECK THIS!)*/;
                }
            }

            // CHECK THIS AGAIN (MOST LIKELY WRONG):
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
            thisState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToCellStates[cell]=_ds[cell]*thisState->forgetGateValues[cell];
            if(!dxcWeightsSet)
                dxcWeightsSet=true;
        }

        // bottom_diff_x:
        memcpy(thisState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToInputs,dxc,inputCount*sizeof(double));
        // bottom_diff_h:
        memcpy(thisState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToOutputs,dxc+inputCount,outputCount*sizeof(double));

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

    double ***gateLayerWeightDiffs;
    double **gateLayerBiasWeightDiffs;

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
                gateLayerWeightDiffs=wf_diff[cell];
                gateLayerBiasWeightDiffs=ibf_diff[cell];
            }
            else if(gate==2)
            {
                // Input gate

                gateLayerWeights=latestState->inputGateLayerWeights[cell];
                gateLayerBiasWeights=latestState->inputGateLayerBiasWeights[cell];
                gateLayerWeightDiffs=wi_diff[cell];
                gateLayerBiasWeightDiffs=ibi_diff[cell];
            }
            else if(gate==3)
            {
                // Output gate

                gateLayerWeights=latestState->outputGateLayerWeights[cell];
                gateLayerBiasWeights=latestState->outputGateLayerBiasWeights[cell];
                gateLayerWeightDiffs=wo_diff[cell];
                gateLayerBiasWeightDiffs=ibo_diff[cell];
            }
            else // if(gate==4)
            {
                // Candidate gate

                gateLayerWeights=latestState->candidateGateLayerWeights[cell];
                gateLayerBiasWeights=latestState->candidateGateLayerBiasWeights[cell];
                gateLayerWeightDiffs=wg_diff[cell];
                gateLayerBiasWeightDiffs=ibg_diff[cell];
            }

            for(uint32_t _currentLayer=hiddenLayerCount+1/*Include topmost output layer*/;_currentLayer>0;_currentLayer--)
            {
                uint32_t currentLayer=_currentLayer-1;
                uint32_t neuronsInThisLayer=currentLayer==hiddenLayerCount/*Is topmost output layer?*/?inputAndOutputCount:hiddenLayerNeuronCounts[currentLayer];
                uint32_t neuronsInPreviousLayer=currentLayer==0?inputAndOutputCount:hiddenLayerNeuronCounts[currentLayer-1];
                for(uint32_t neuronInThisLayer=0;neuronInThisLayer<neuronsInThisLayer;neuronInThisLayer++)
                {
                    // Adjust bias of this neuron
                    gateLayerBiasWeights[currentLayer][neuronInThisLayer]-=learningRate*gateLayerBiasWeightDiffs[currentLayer][neuronInThisLayer];
                    for(uint32_t neuronInPreviousLayer=0;neuronInPreviousLayer<neuronsInPreviousLayer;neuronInPreviousLayer++)
                    {
                        // Adjust weight from neuronInPreviousLayer to neuronInThisLayer
                        gateLayerWeights[currentLayer][neuronInThisLayer][neuronInPreviousLayer]-=learningRate*gateLayerWeightDiffs[currentLayer][neuronInThisLayer][neuronInPreviousLayer];
                    }
                    free(gateLayerWeightDiffs[currentLayer][neuronInThisLayer]);
                }
                free(gateLayerBiasWeightDiffs[currentLayer]);
                free(gateLayerWeightDiffs[currentLayer]);
                if(gate==4)
                {
                    // Final gate; free error terms
                    free(i_errorTerms[cell][currentLayer]);
                    free(f_errorTerms[cell][currentLayer]);
                    free(o_errorTerms[cell][currentLayer]);
                    free(g_errorTerms[cell][currentLayer]);
                }
            }
            free(gateLayerBiasWeightDiffs);
            free(gateLayerWeightDiffs);
        }

        // Free error terms
        free(i_errorTerms[cell]);
        free(f_errorTerms[cell]);
        free(o_errorTerms[cell]);
        free(g_errorTerms[cell]);

        latestState->inputGateBiasWeights[cell]-=learningRate*bi_diff[cell];
        latestState->forgetGateBiasWeights[cell]-=learningRate*bf_diff[cell];
        latestState->outputGateBiasWeights[cell]-=learningRate*bo_diff[cell];
        latestState->candidateGateBiasWeights[cell]-=learningRate*bg_diff[cell];

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

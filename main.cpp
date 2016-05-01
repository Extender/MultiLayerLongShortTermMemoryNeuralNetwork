#include <stdlib.h>
#include <iostream>
#include <stdint.h>

#include "io.h"
#include "text.h"

#include <windows.h>

#include "lstmlayer.h"

using namespace std;

string doubleArrayToString(double *array,uint32_t elementCount,bool includeHighest)
{
    string out;
    out+="[";
    double highest=-9999.99;
    uint32_t highestIndex;
    for(uint32_t i=0;i<elementCount;i++)
    {
        if(i>0)
            out+=", ";
        char *str=text::doubleToString(array[i]);
        if(includeHighest&&array[i]>highest)
        {
            highest=array[i];
            highestIndex=i;
        }
        out+=str;
        free(str);
    }
    if(includeHighest&&elementCount>0)
    {
        out+=" (highest: ";
        char *highestStr=text::unsignedIntToString(highestIndex);
        out+=highestStr;
        out+=")";
        free(highestStr);
    }
    out+="]";
    return out;
}

int main(int argc, char *argv[])
{
    /*
    See:
    http://nicodjimenez.github.io/2014/08/08/lstm.html

    https://github.com/nicodjimenez/lstm/blob/master/lstm.py
    https://github.com/nicodjimenez/lstm/blob/master/test.py

    lstm_param: same for all layers!
    lstm_state: different for every layer!

    Is there a problem with the learning method we are using in main()?


    y_list: list of desired outputs of states
    x_list: list of inputs of states
    lstm_node_list: list of parameters (weights, biases, gate values) of states
    In Nico's example, only one output is needed, so he only sets pred[0] and diff[0] in ToyLossLayer.

    */

    char *helloString="hello";
    uint32_t inputCount=3; // h, e, l
    uint32_t outputCount=3; // e, l, o
    uint32_t backpropagationSteps=3;
    double learningRate=0.2;
    LSTMLayer *lstm=new LSTMLayer(inputCount,outputCount,backpropagationSteps,learningRate,2);
    uint64_t cycle=0;
    // Perform an alphabet shift (+2 letters)
    char *str;
    double **desiredOutputs=(double**)malloc((backpropagationSteps+1)*sizeof(double*));
    // Only call learn() after the last step!
    for(uint64_t current=0;/*current<10*/;current++)
    {
        double *input=(double*)malloc(inputCount*sizeof(double));
        uint64_t currentPos=current%4; // o not used!
        if(currentPos==0)
        {
            str=text::unsignedLongToString(cycle);
            cout<<"*** New cycle (#"<<str<<") ***"<<endl<<endl;
            free(str);
        }
        uint8_t currentChar;
        if(currentPos==0)
            currentChar=0;
        else if(currentPos==1)
            currentChar=1;
        else // if(currentPos==2||currentPos==3)
            currentChar=2;
        str=text::unsignedLongToString(currentPos);
        cout<<"Current position: "<<str<<" ("<<helloString[currentPos]<<")"<<endl;
        free(str);
        for(uint32_t i=0;i<inputCount;i++)
            input[i]=(i==currentChar?1.0:0.0);
        double *output;
        output=lstm->process(input);
        cout<<"Output:           "<<doubleArrayToString(output,outputCount,true)<<endl;
        free(output);
        free(input);

        double *desiredOutput=(double*)malloc(outputCount*sizeof(double));
        // Desired output: next char!
        uint8_t desiredOut;
        if(currentPos==0) // "h"
            desiredOut=0; // => e
        else if(currentPos==1) // "e"
            desiredOut=1; // => l
        else if(currentPos==2) // "l"
            desiredOut=1; // => l
        else // if(currentPos==3) // "l"
            desiredOut=2; // => o
        for(uint32_t i=0;i<outputCount;i++)
            desiredOutput[i]=(i==desiredOut?1.0:0.0);

        cout<<"Desired output:   "<<doubleArrayToString(desiredOutput,outputCount,false)<<endl;

        desiredOutputs[currentPos]=desiredOutput;

        if(currentPos==3)
        {
            lstm->learn(desiredOutputs);
            free(desiredOutputs[0]);
            free(desiredOutputs[1]);
            free(desiredOutputs[2]);
            free(desiredOutputs[3]);
            cycle++;
        }


        cout<<endl;
        //Sleep(200);
        //if(current%1000==0)
        //    system("cls");
    }
    free(desiredOutputs);
    delete lstm;
}


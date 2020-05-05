#ifndef LSTMNETWORK_H_
#define LSTMNETWORK_H_

#include <vector>
#include "MemoryBlock.cuh"
#include "Neuron.cuh"
#include <cuda.h>
using namespace std;

__global__ void forwardPass(Neuron **neurons, double *connections, double *activations, int size);
__global__ void backwardPass(Neuron **neurons, double *weightedError, double *errorSum, double learningRate, int connections);
__global__ void forwardPassLSTM(MemoryBlock *block, double **connections, double **activations, int size, int cycles);
__global__ void backwardPassLSTM(MemoryBlock *block, double **weightedError, double *errorSum, double learningRate, int connections, int cycles);


class TextClassifier {
private:
	const int maxBlocks = 1;
	const int maxThreads = 1;
	unsigned int inputSize;
	double learningRate;
	MemoryBlock *block;
	vector<Neuron> logits_layer;
//	vector<double> timeSteps;
public:
    TextClassifier(int is, int c, double lr, int num_classes);
	virtual ~TextClassifier();
//	vector<double> classify(vector<double> input);
	vector<double> train(Vector<vector<double>> input, vector<double> target);
};

#endif /* LSTMNETWORK_H_ */

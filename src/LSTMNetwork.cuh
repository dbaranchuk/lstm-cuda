#ifndef LSTMNETWORK_H_
#define LSTMNETWORK_H_

#include <vector>
#include "MemoryBlock.cuh"
#include "Neuron.cuh"
#include <cuda.h>
using namespace std;

__global__ void forwardPass(Neuron **neurons, double *connections, double *activations, int size, int cycles);
__global__ void backwardPass(Neuron **neurons, double *weightedError, double *errorSum, double learningRate, int connections, int size, int cycles);
__global__ void forwardPassLSTM(MemoryBlock **blocks, double *connections, double *activations, int size, int cycles);
__global__ void backwardPassLSTM(MemoryBlock **blocks, double **weightedError, double *errorSum, double learningRate, int connections, int size, int cycles);


class LSTMNetwork {
private:
	const int maxBlocks = 256;
	const int maxThreads = 256;
	unsigned int inputSize;
	double learningRate;
	vector<MemoryBlock> blocks;
	vector<Neuron> layer;
	vector<double> timeSteps;
public:
	LSTMNetwork(int is, int b, int c, double l, int num_classes);
	virtual ~LSTMNetwork();
	vector<double> classify(vector<double> input);
	vector<double> train(vector<double> input, vector<double> target);
	vector<double> forward(vector<double> input);
};

#endif /* LSTMNETWORK_H_ */

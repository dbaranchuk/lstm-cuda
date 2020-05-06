#ifndef TextClassifier_H_
#define TextClassifier_H_

#include <vector>
#include "LSTMCell.cuh"
#include "Neuron.cuh"
#include <cuda.h>
#include <math.h>

using namespace std;

__global__ void logits_forward_pass(Neuron **neurons, double *connections, double *activations, int size);
//__global__ void backwardPass(Neuron **neurons, double *weightedError, double *errorSum, double learningRate, int connections);
__global__ void lstm_forward_pass(MemoryBlock *block, double *connections, double **activations, int size);
//__global__ void backwardPassLSTM(MemoryBlock *block, double **weightedError, double *errorSum, double learningRate, int connections, int cycles);


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
	double train(vector<double> &input, vector<double> &target);
};

#endif /* TextClassifier_H_ */

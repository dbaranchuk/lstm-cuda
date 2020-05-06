#ifndef TextClassifier_H_
#define TextClassifier_H_

#include <vector>
#include "LSTMCell.cuh"
#include "Neuron.cuh"
#include <cuda.h>
#include <math.h>

using namespace std;

__global__ void logits_forward_pass(Neuron **neurons, double *connections, double *activations, int size);
__global__ void lstm_forward_pass(LSTMCell *block, double *connections, double **activations, int size);


class TextClassifier {
private:
	const int maxBlocks = 1;
	const int maxThreads = 1;
	unsigned int inputSize;
	double learningRate;
	LSTMCell *block;
	vector<Neuron> logits_layer;
public:
    TextClassifier(int is, int c, double lr, int num_classes);
	virtual ~TextClassifier();
	double train(vector<double> &input, vector<double> &target);
};

#endif /* TextClassifier_H_ */

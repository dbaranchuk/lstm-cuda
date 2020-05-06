#ifndef LSTMCell_H_
#define LSTMCell_H_

#include "MemoryCell.cuh"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <random>
using namespace std;

class LSTMCell {
private:
	static long long n;
	__device__ double sigmoid(double input);
	__device__ double sigmoidPrime(double input);
public:
	int nConnections;
	int nCells;
	MemoryCell **cells;
	double *input_data_weight,
		*forget_data_weight, *output_data_weight, *bias,
		//**impulses,
		*input_hidden_weight,
		*forget_hidden_weight,
		*output_hidden_weight;
	double input, inputPrime,
		forget, forgetPrime,
		output, outputPrime;
	__device__ double inputGate(double data);
	__device__ double forgetGate(double data);
	__device__ double outputGate(double data);

	LSTMCell(int output_size, int hidden_size);
	virtual ~LSTMCell();
	static LSTMCell *copyToGPU(LSTMCell *memory);
	static LSTMCell *copyFromGPU(LSTMCell *memory);
};

#endif /* LSTMCell_H_ */

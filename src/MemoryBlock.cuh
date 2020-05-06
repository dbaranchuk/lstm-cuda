#ifndef MEMORYBLOCK_H_
#define MEMORYBLOCK_H_

#include "MemoryCell.cuh"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <random>
using namespace std;

class MemoryBlock {
private:
	static long long n;
	__device__ double sigmoid(double input);
	__device__ double sigmoidPrime(double input);
public:
	int nConnections;
	int nCells;
	MemoryCell **cells;
	double *input_data_weight,
		*forget_data_weight, *output_data_weight,
		*bias, //**impulses,
		*input_hidden_weight,
		*forget_hidden_weight,
		*output_hidden_weight;
	double input, inputPrime,
		forget, forgetPrime,
		output, outputPrime;
	__device__ double inputGate(double data);
	__device__ double forgetGate(double data);
	__device__ double outputGate(double data);
	MemoryBlock(int cl, int cn);
	virtual ~MemoryBlock();
	__device__ double *forward(double *input);
//	__device__ double *backward(double *errorPrime, double learningRate);
	static MemoryBlock *copyToGPU(MemoryBlock *memory);
	static MemoryBlock *copyFromGPU(MemoryBlock *memory);
};

#endif /* MEMORYBLOCK_H_ */

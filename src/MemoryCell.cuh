#ifndef MEMORYCELL_H_
#define MEMORYCELL_H_

#include "BaseNode.cuh"
#include <vector>
#include <math.h>
#include <time.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <random>
using namespace std;

class MemoryCell : BaseNode {
private:
	static long long n;
public:
	int nConnections;
	double *cell_data_weight, *cell_data_partial,
		*input_data_partial, *forget_data_partial;
	double cell_hidden_weight, bias;
	double activationIn, activationInPrime,
		activationOut, activationOutPrime,
		state, previousState,
		feedback, previousFeedback,
		cell_hidden_partial;
	double input_hidden_partial,
		forget_hidden_partial;
	__device__ double activateIn(double data);
	__device__ double activateOut(double data);
	MemoryCell(int c);
	virtual ~MemoryCell();
	static MemoryCell *copyToGPU(MemoryCell *memory);
	static MemoryCell *copyFromGPU(MemoryCell *memory);
};

#endif /* MEMORYCELL_H_ */

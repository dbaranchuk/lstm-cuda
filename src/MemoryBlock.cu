#include "MemoryBlock.cuh"

long long int MemoryBlock::n = 0;

MemoryBlock::MemoryBlock(int cl, int cn) {
	nConnections = cn;
	nCells = cl;
	input = 0; inputPrime = 0;
	forget = 0; forgetPrime = 0;
	output = 0; outputPrime = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	bias = (double *)calloc(3, sizeof(double));
	cells = (MemoryCell **)malloc(sizeof(MemoryCell *) * nCells);
	inputFeedbackWeight = (double *)malloc(sizeof(double) * nCells);
	forgetFeedbackWeight = (double *)malloc(sizeof(double) * nCells);
	outputFeedbackWeight = (double *)malloc(sizeof(double) * nCells);

	for (int i = 0; i < nCells; i++) {
		cells[i] = (new MemoryCell(nConnections));
		inputFeedbackWeight[i] = (d(g));
		forgetFeedbackWeight[i] = (d(g));
		outputFeedbackWeight[i] = (d(g));
	}

	impulse = (double *)malloc(sizeof(double) * nConnections);
	inputDataWeight = (double *)malloc(sizeof(double) * nConnections);
	forgetDataWeight = (double *)malloc(sizeof(double) * nConnections);
	outputDataWeight = (double *)malloc(sizeof(double) * nConnections);

	for (int i = 0; i < nConnections; i++) {
		impulse[i] = (0);
		inputDataWeight[i] = (d(g));
		forgetDataWeight[i] = (d(g));
		outputDataWeight[i] = (d(g));
	}
}

MemoryBlock::~MemoryBlock() {
}


__device__ double MemoryBlock::inputGate(double data) {
	input = sigmoid(data);
	inputPrime = sigmoidPrime(data);
	return input;
}

__device__ double MemoryBlock::forgetGate(double data) {
	forget = sigmoid(data);
	forgetPrime = sigmoidPrime(data);
	return forget;
}

__device__ double MemoryBlock::outputGate(double data) {
	output = sigmoid(data);
	outputPrime = sigmoidPrime(data);
	return output;
}

__device__ double *MemoryBlock::forward(double *input) {
	double *cellSum = new double[nCells] {0};
	double inputSum = bias[0];
	double forgetSum = bias[1];
	double outputSum = bias[2];

	for (int i = 0; i < nCells; i++) {
		inputSum += (inputFeedbackWeight[i] * cells[i]->feedback);
		forgetSum += (forgetFeedbackWeight[i] * cells[i]->feedback);
		outputSum += (outputFeedbackWeight[i] * cells[i]->feedback);
	}

	// find the weighted sum of all input
	for (int i = 0; i < nConnections; i++) {
		for (unsigned int j = 0; j < nCells; j++) {
			cellSum[j] += input[i] * cells[j]->cellDataWeight[i];
		}
		inputSum += input[i] * inputDataWeight[i];
		forgetSum += input[i] * forgetDataWeight[i];
		outputSum += input[i] * outputDataWeight[i];
	}

	// compute input into memory
	double *output = new double[nCells];	// potential error
	for (int i = 0; i < nCells; i++) {
		cells[i]->previousState = cells[i]->state;
		cells[i]->state *= forgetGate(forgetSum);
		cells[i]->state += cells[i]->activateIn(cellSum[i]) * inputGate(inputSum);

		// compute output of memory cell
		cells[i]->previousFeedback = cells[i]->feedback;
		cells[i]->feedback = cells[i]->activateOut(cells[i]->state) * outputGate(outputSum);
		output[i] = (cells[i]->feedback);
	}

	return output;
}

// errorprime must be a vector with length of number of cells
__device__ double *MemoryBlock::backward(double *errorPrime, double learningRate) {
	double *eta = new double[nCells],
			*inputDataPartialSum = new double[nConnections] {0},
			*forgetDataPartialSum = new double[nConnections] {0};
	double blockSum = 0,
			inputFeedbackPartialSum = 0,
			forgetFeedbackPartialSum = 0;

	for (int i = 0; i < nCells; i++) {
		blockSum += cells[i]->activationOut * errorPrime[i];
		eta[i] = (output * cells[i]->activationOutPrime * errorPrime[i]);
		outputFeedbackWeight[i] -= learningRate * blockSum * outputPrime * cells[i]->feedback;
	}

	for (int i = 0; i < nConnections; i++) {
		outputDataWeight[i] -= learningRate * blockSum * outputPrime * impulse[i];	// invalid read of size 8
	}

	// calculate the updates, and update the cell weights
	for (int i = 0; i < nCells; i++) {
		for (int j = 0; j < nConnections; j++) {
			cells[i]->cellDataPartial[j] = cells[i]->cellDataPartial[j] * forget + cells[i]->activationInPrime * input * impulse[j];
			cells[i]->cellDataWeight[j] -= learningRate * eta[i] * cells[i]->cellDataPartial[j];
			cells[i]->forgetDataPartial[j] = cells[i]->forgetDataPartial[j] * forget + cells[i]->previousState * forgetPrime * impulse[j];	// invalid read of size 8
			cells[i]->inputDataPartial[j] = cells[i]->inputDataPartial[j] * forget + cells[i]->activationIn * inputPrime * impulse[j];	// invalid read of size 8
			forgetDataPartialSum[j] += cells[i]->forgetDataPartial[j] * eta[i];
			inputDataPartialSum[j] += cells[i]->inputDataPartial[j] * eta[i];
		}

		cells[i]->cellFeedbackPartial = cells[i]->cellFeedbackPartial * forget + cells[i]->activationInPrime * input * cells[i]->previousFeedback;
		cells[i]->cellFeedbackWeight -= learningRate * eta[i] * cells[i]->cellFeedbackPartial;

		cells[i]->forgetFeedbackPartial = cells[i]->forgetFeedbackPartial * forget + cells[i]->previousState * forgetPrime * cells[i]->previousFeedback;
		forgetFeedbackPartialSum += eta[i] *cells[i]->forgetFeedbackPartial;

		cells[i]->inputFeedbackPartial = cells[i]->inputFeedbackPartial * forget + cells[i]->activationIn * inputPrime * cells[i]->previousFeedback;
		inputFeedbackPartialSum += eta[i] *cells[i]->inputFeedbackPartial;
	}

	// update the input, output, and forget weights
	for (int i = 0; i < nCells; i++) {
		for (int j = 0; j < nConnections; j++) {
			forgetDataWeight[j] -= learningRate * forgetDataPartialSum[j];	// invalid read of size 8
			inputDataWeight[j] -= learningRate * inputDataPartialSum[j];	// invalid read of size 8
		}
		inputFeedbackWeight[i] -= learningRate * inputFeedbackPartialSum;
		forgetFeedbackWeight[i] -= learningRate * forgetFeedbackPartialSum;
	}

	double *temp = new double[nConnections];	// potential error
	for (int i = 0; i < nConnections; i++) {
		temp[i] = (0.0);
	}


	return temp;
}

MemoryBlock *MemoryBlock::copyToGPU(MemoryBlock *memory) {
	MemoryBlock *memoryBlock;
	cudaMalloc((void **)&memoryBlock, (sizeof(MemoryBlock)));
	cudaDeviceSynchronize();
	cudaMemcpy(memoryBlock, memory, sizeof(MemoryBlock), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	MemoryCell **memoryCells;
	cudaMalloc((void **)&memoryCells, ((sizeof(MemoryCell *) * memory->nCells)));
	for (int i = 0; i < memory->nCells; i++) {
		MemoryCell *buffer = MemoryCell::copyToGPU(memory->cells[i]);
		cudaMemcpy(&memoryCells[i], &buffer, sizeof(MemoryCell *), cudaMemcpyHostToDevice);
	} cudaMemcpy(&(memoryBlock->cells), &memoryCells, sizeof(MemoryCell **), cudaMemcpyHostToDevice);


	double *ifw, *ffw, *ofw, *b;
	cudaMalloc((void **)&ifw, (sizeof(double) * memory->nCells));
	cudaMalloc((void **)&ffw, (sizeof(double) * memory->nCells));
	cudaMalloc((void **)&ofw, (sizeof(double) * memory->nCells));
	cudaMalloc((void **)&b, (sizeof(double) * 3));

	double *idw, *fdw, *odw, *i;
	cudaMalloc((void **)&idw, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&fdw, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&odw, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&i, (sizeof(double) * memory->nConnections));
	cudaDeviceSynchronize();

	cudaMemcpy(ifw, memory->inputFeedbackWeight, (sizeof(double) * memory->nCells), cudaMemcpyHostToDevice);
	cudaMemcpy(ffw, memory->forgetFeedbackWeight, (sizeof(double) * memory->nCells), cudaMemcpyHostToDevice);
	cudaMemcpy(ofw, memory->outputFeedbackWeight, (sizeof(double) * memory->nCells), cudaMemcpyHostToDevice);
	cudaMemcpy(b, memory->bias, (sizeof(double) * 3), cudaMemcpyHostToDevice);
	cudaMemcpy(idw, memory->inputDataWeight, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(fdw, memory->forgetDataWeight, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(odw, memory->outputDataWeight, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(i, memory->impulse, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(&(memoryBlock->inputFeedbackWeight), &(ifw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->forgetFeedbackWeight), &(ffw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->outputFeedbackWeight), &(ofw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->bias), &(b), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->inputDataWeight), &(idw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->forgetDataWeight), &(fdw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->outputDataWeight), &(odw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->impulse), &(i), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	return memoryBlock;
}

MemoryBlock *MemoryBlock::copyFromGPU(MemoryBlock *memory) {

	MemoryBlock *memoryBlock;
	memoryBlock = (MemoryBlock *)malloc((sizeof(MemoryBlock)));
	cudaDeviceSynchronize();
	cudaMemcpy(memoryBlock, memory, sizeof(MemoryBlock), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	MemoryCell **memoryCells;
	memoryCells = (MemoryCell **)malloc((sizeof(MemoryCell *) * memoryBlock->nCells));
	cudaMemcpy(memoryCells, memoryBlock->cells, (sizeof(MemoryCell *) * memoryBlock->nCells), cudaMemcpyDeviceToHost);

	for (int i = 0; i < memoryBlock->nCells; i++) {
		MemoryCell *buffer = MemoryCell::copyFromGPU(memoryCells[i]);
		memoryCells[i] = buffer;
	} memcpy(&(memoryBlock->cells), &memoryCells, sizeof(MemoryCell *));


	double *ifw, *ffw, *ofw, *b;
	ifw = (double *)malloc((sizeof(double) * memoryBlock->nCells));
	ifw = (double *)malloc((sizeof(double) * memoryBlock->nCells));
	ffw = (double *)malloc((sizeof(double) * memoryBlock->nCells));
	ofw = (double *)malloc((sizeof(double) * memoryBlock->nCells));
	b = (double *)malloc((sizeof(double) * 3));

	double *idw, *fdw, *odw, *i;
	idw = (double *)malloc((sizeof(double) * memoryBlock->nConnections));
	fdw = (double *)malloc((sizeof(double) * memoryBlock->nConnections));
	odw = (double *)malloc((sizeof(double) * memoryBlock->nConnections));
	i = (double *)malloc((sizeof(double) * memoryBlock->nConnections));
	cudaDeviceSynchronize();

	cudaMemcpy(ifw, memoryBlock->inputFeedbackWeight, (sizeof(double) * memoryBlock->nCells), cudaMemcpyDeviceToHost);
	cudaMemcpy(ffw, memoryBlock->forgetFeedbackWeight, (sizeof(double) * memoryBlock->nCells), cudaMemcpyDeviceToHost);
	cudaMemcpy(ofw, memoryBlock->outputFeedbackWeight, (sizeof(double) * memoryBlock->nCells), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, memoryBlock->bias, (sizeof(double) * 3), cudaMemcpyDeviceToHost);
	cudaMemcpy(idw, memoryBlock->inputDataWeight, (sizeof(double) * memoryBlock->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(fdw, memoryBlock->forgetDataWeight, (sizeof(double) * memoryBlock->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(odw, memoryBlock->outputDataWeight, (sizeof(double) * memoryBlock->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(i, memoryBlock->impulse, (sizeof(double) * memoryBlock->nConnections), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	memcpy(&(memoryBlock->inputFeedbackWeight), &ifw, (sizeof(double *)));
	memcpy(&(memoryBlock->forgetFeedbackWeight), &ffw, (sizeof(double *)));
	memcpy(&(memoryBlock->outputFeedbackWeight), &ofw, (sizeof(double *)));
	memcpy(&(memoryBlock->bias), &b, (sizeof(double *)));
	memcpy(&(memoryBlock->inputDataWeight), &idw, (sizeof(double *)));
	memcpy(&(memoryBlock->forgetDataWeight), &fdw, (sizeof(double *)));
	memcpy(&(memoryBlock->outputDataWeight), &odw, (sizeof(double *)));
	memcpy(&(memoryBlock->impulse), &i, (sizeof(double *)));

	return memoryBlock;
}


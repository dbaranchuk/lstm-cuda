#include "LSTMCell.cuh"

long long int LSTMCell::n = 0;

LSTMCell::LSTMCell(int output_size, int input_size) {
	nConnections = input_size;
	nCells = output_size;
	input = 0; inputPrime = 0;
	forget = 0; forgetPrime = 0;
	output = 0; outputPrime = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	bias = (double *)calloc(3, sizeof(double));
	cells = (MemoryCell **)malloc(sizeof(MemoryCell *) * nCells);
	input_hidden_weight = (double *)malloc(sizeof(double) * nCells);
	forget_hidden_weight = (double *)malloc(sizeof(double) * nCells);
	output_hidden_weight = (double *)malloc(sizeof(double) * nCells);

	for (int i = 0; i < nCells; i++) {
		cells[i] = (new MemoryCell(output_size));
		input_hidden_weight[i] = d(g);
		forget_hidden_weight[i] = d(g);
		output_hidden_weight[i] = d(g);
	}

	//impulse = (double *)malloc(sizeof(double) * nConnections);
	input_data_weight = (double *)malloc(sizeof(double) * nConnections);
	forget_data_weight = (double *)malloc(sizeof(double) * nConnections);
	output_data_weight = (double *)malloc(sizeof(double) * nConnections);

	for (int i = 0; i < nConnections; i++) {
		//impulse[i] = 0;
		input_data_weight[i] = d(g);
		forget_data_weight[i] = d(g);
		output_data_weight[i] = d(g);
	}
}

LSTMCell::~LSTMCell() {
}


__device__ double LSTMCell::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

__device__ double LSTMCell::sigmoidPrime(double input) {
	return sigmoid(input) * (1 - sigmoid(input));
}

__device__ double LSTMCell::inputGate(double data) {
	input = sigmoid(data);
	inputPrime = sigmoidPrime(data);
	return input;
}

__device__ double LSTMCell::forgetGate(double data) {
	forget = sigmoid(data);
	forgetPrime = sigmoidPrime(data);
	return forget;
}

__device__ double LSTMCell::outputGate(double data) {
	output = sigmoid(data);
	outputPrime = sigmoidPrime(data);
	return output;
}

__device__ double *LSTMCell::forward(double *input) {
	double *cellSum = new double[nCells] {0};
	double inputSum = bias[0];
	double forgetSum = bias[1];
	double outputSum = bias[2];

	for (int i = 0; i < nCells; i++) {
		inputSum += input_hidden_weight[i] * cells[i]->feedback;
		forgetSum += forget_hidden_weight[i] * cells[i]->feedback;
		outputSum += output_hidden_weight[i] * cells[i]->feedback;
	}

	// find the weighted sum of all input
	for (int i = 0; i < nConnections; i++) {
		for (unsigned int j = 0; j < nCells; j++) {
			cellSum[j] += input[i] * cells[j]->cell_data_weight[i];
		}
		inputSum += input[i] * input_data_weight[i];
		forgetSum += input[i] * forget_data_weight[i];
		outputSum += input[i] * output_data_weight[i];
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
		output[i] = cells[i]->feedback;
	}

	return output;
}

// errorprime must be a vector with length of number of cells
//__device__ double *LSTMCell::backward(double *errorPrime, double learningRate) {
//	double *eta = new double[nCells],
//			*inputDataPartialSum = new double[nConnections] {0},
//			*forgetDataPartialSum = new double[nConnections] {0};
//	double blockSum = 0,
//			inputFeedbackPartialSum = 0,
//			forgetFeedbackPartialSum = 0;
//
//	for (int i = 0; i < nCells; i++) {
//		blockSum += cells[i]->activationOut * errorPrime[i];
//		eta[i] = (output * cells[i]->activationOutPrime * errorPrime[i]);
//		output_hidden_weight[i] -= learningRate * blockSum * outputPrime * cells[i]->feedback;
//	}
//
//	for (int i = 0; i < nConnections; i++) {
//		output_data_weight[i] -= learningRate * blockSum * outputPrime * impulse[i];	// invalid read of size 8
//	}
//
//	// calculate the updates, and update the cell weights
//	for (int i = 0; i < nCells; i++) {
//		for (int j = 0; j < nConnections; j++) {
//			cells[i]->cellDataPartial[j] = cells[i]->cellDataPartial[j] * forget + cells[i]->activationInPrime * input * impulse[j];
//			cells[i]->cell_data_weight[j] -= learningRate * eta[i] * cells[i]->cellDataPartial[j];
//			cells[i]->forgetDataPartial[j] = cells[i]->forgetDataPartial[j] * forget + cells[i]->previousState * forgetPrime * impulse[j];	// invalid read of size 8
//			cells[i]->inputDataPartial[j] = cells[i]->inputDataPartial[j] * forget + cells[i]->activationIn * inputPrime * impulse[j];	// invalid read of size 8
//			forgetDataPartialSum[j] += cells[i]->forgetDataPartial[j] * eta[i];
//			inputDataPartialSum[j] += cells[i]->inputDataPartial[j] * eta[i];
//		}
//
//		cells[i]->cellFeedbackPartial = cells[i]->cellFeedbackPartial * forget + cells[i]->activationInPrime * input * cells[i]->previousFeedback;
//		cells[i]->cell_hidden_weight -= learningRate * eta[i] * cells[i]->cellFeedbackPartial;
//
//		cells[i]->forgetFeedbackPartial = cells[i]->forgetFeedbackPartial * forget + cells[i]->previousState * forgetPrime * cells[i]->previousFeedback;
//		forgetFeedbackPartialSum += eta[i] *cells[i]->forgetFeedbackPartial;
//
//		cells[i]->inputFeedbackPartial = cells[i]->inputFeedbackPartial * forget + cells[i]->activationIn * inputPrime * cells[i]->previousFeedback;
//		inputFeedbackPartialSum += eta[i] *cells[i]->inputFeedbackPartial;
//	}
//
//	// update the input, output, and forget weights
//	for (int i = 0; i < nCells; i++) {
//		for (int j = 0; j < nConnections; j++) {
//			forget_data_weight[j] -= learningRate * forgetDataPartialSum[j];	// invalid read of size 8
//			input_data_weight[j] -= learningRate * inputDataPartialSum[j];	// invalid read of size 8
//		}
//		input_hidden_weight[i] -= learningRate * inputFeedbackPartialSum;
//		forget_hidden_weight[i] -= learningRate * forgetFeedbackPartialSum;
//	}
//
//	double *temp = new double[nConnections];	// potential error
//	for (int i = 0; i < nConnections; i++) {
//		temp[i] = (0.0);
//	}
//
//	return temp;
//}

LSTMCell *LSTMCell::copyToGPU(LSTMCell *memory) {
	LSTMCell *memoryBlock;
	cudaMalloc((void **)&memoryBlock, (sizeof(LSTMCell)));
	cudaDeviceSynchronize();
	cudaMemcpy(memoryBlock, memory, sizeof(LSTMCell), cudaMemcpyHostToDevice);
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

	cudaMemcpy(ifw, memory->input_hidden_weight, (sizeof(double) * memory->nCells), cudaMemcpyHostToDevice);
	cudaMemcpy(ffw, memory->forget_hidden_weight, (sizeof(double) * memory->nCells), cudaMemcpyHostToDevice);
	cudaMemcpy(ofw, memory->output_hidden_weight, (sizeof(double) * memory->nCells), cudaMemcpyHostToDevice);
	cudaMemcpy(b, memory->bias, (sizeof(double) * 3), cudaMemcpyHostToDevice);
	cudaMemcpy(idw, memory->input_data_weight, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(fdw, memory->forget_data_weight, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(odw, memory->output_data_weight, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	//cudaMemcpy(i, memory->impulse, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(&(memoryBlock->input_hidden_weight), &(ifw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->forget_hidden_weight), &(ffw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->output_hidden_weight), &(ofw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->bias), &(b), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->input_data_weight), &(idw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->forget_data_weight), &(fdw), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryBlock->output_data_weight), &(odw), (sizeof(double *)), cudaMemcpyHostToDevice);
	//cudaMemcpy(&(memoryBlock->impulse), &(i), (sizeof(double *)), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	return memoryBlock;
}

LSTMCell *LSTMCell::copyFromGPU(LSTMCell *memory) {

	LSTMCell *memoryBlock;
	memoryBlock = (LSTMCell *)malloc((sizeof(LSTMCell)));
	cudaDeviceSynchronize();
	cudaMemcpy(memoryBlock, memory, sizeof(LSTMCell), cudaMemcpyDeviceToHost);
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

	cudaMemcpy(ifw, memoryBlock->input_hidden_weight, (sizeof(double) * memoryBlock->nCells), cudaMemcpyDeviceToHost);
	cudaMemcpy(ffw, memoryBlock->forget_hidden_weight, (sizeof(double) * memoryBlock->nCells), cudaMemcpyDeviceToHost);
	cudaMemcpy(ofw, memoryBlock->output_hidden_weight, (sizeof(double) * memoryBlock->nCells), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, memoryBlock->bias, (sizeof(double) * 3), cudaMemcpyDeviceToHost);
	cudaMemcpy(idw, memoryBlock->input_data_weight, (sizeof(double) * memoryBlock->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(fdw, memoryBlock->forget_data_weight, (sizeof(double) * memoryBlock->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(odw, memoryBlock->output_data_weight, (sizeof(double) * memoryBlock->nConnections), cudaMemcpyDeviceToHost);
	//cudaMemcpy(i, memoryBlock->impulse, (sizeof(double) * memoryBlock->nConnections), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	memcpy(&(memoryBlock->input_hidden_weight), &ifw, (sizeof(double *)));
	memcpy(&(memoryBlock->forget_hidden_weight), &ffw, (sizeof(double *)));
	memcpy(&(memoryBlock->output_hidden_weight), &ofw, (sizeof(double *)));
	memcpy(&(memoryBlock->bias), &b, (sizeof(double *)));
	memcpy(&(memoryBlock->input_data_weight), &idw, (sizeof(double *)));
	memcpy(&(memoryBlock->forget_data_weight), &fdw, (sizeof(double *)));
	memcpy(&(memoryBlock->output_data_weight), &odw, (sizeof(double *)));
	//memcpy(&(memoryBlock->impulse), &i, (sizeof(double *)));

	return memoryBlock;
}


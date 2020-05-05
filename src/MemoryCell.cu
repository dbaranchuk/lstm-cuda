#include "MemoryCell.cuh"

long long MemoryCell::n = 0;

MemoryCell::MemoryCell(int c) {
	nConnections = c;
	activationIn = 0; activationInPrime = 0;
	activationOut = 0; activationOutPrime = 0;
	state = 0; previousState = 0;
	feedback = 0; previousFeedback = 0;
	bias = 0;

	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);

	cell_hidden_weight = d(g);
	cell_hidden_partial = 0;
	input_hidden_partial = 0;
	forget_hidden_partial = 0;

	cell_data_weight = (double *)malloc(sizeof(double) * c);
	cell_data_partial = (double *)malloc(sizeof(double) * c);
	forget_data_partial = (double *)malloc(sizeof(double) * c);
	input_data_partial = (double *)malloc(sizeof(double) * c);

	for (int i = 0; i < c; i++) {
		cell_data_weight[i] = (d(g));
		cell_data_partial[i] = (0);
		forget_data_partial[i] = (0);
		input_data_partial[i] = (0);
	}
}

MemoryCell::~MemoryCell() {}

__device__ double MemoryCell::activateIn(double data) {
	activationIn = activationFunction(data);
	activationInPrime = activationFunctionPrime(data);
	return activationIn;
}

__device__ double MemoryCell::activateOut(double data) {
	activationOut = activationFunction(data);
	activationOutPrime = activationFunctionPrime(data);
	return activationOut;
}

MemoryCell *MemoryCell::copyToGPU(MemoryCell *memory) {
	MemoryCell *memoryCell;
	cudaMalloc((void **)&memoryCell, (sizeof(MemoryCell)));
	cudaDeviceSynchronize();
	cudaMemcpy(memoryCell, memory, sizeof(MemoryCell), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	double *cdw, *idp, *fdp, *cdp;
	cudaMalloc((void **)&cdw, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&idp, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&fdp, (sizeof(double) * memory->nConnections));
	cudaMalloc((void **)&cdp, (sizeof(double) * memory->nConnections));
	cudaDeviceSynchronize();

	cudaMemcpy(cdw, memory->cell_data_weight, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(idp, memory->input_data_partial, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(fdp, memory->forget_data_partial, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaMemcpy(cdp, memory->cell_data_partial, (sizeof(double) * memory->nConnections), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(&(memoryCell->cell_data_weight), &cdw, sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryCell->input_data_partial), &idp, sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryCell->forget_data_partial), &fdp, sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(memoryCell->cell_data_partial), &cdp, sizeof(double *), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	return memoryCell;
}

MemoryCell *MemoryCell::copyFromGPU(MemoryCell *memory) {

	MemoryCell *memoryCell;
	memoryCell = (MemoryCell *)malloc((sizeof(MemoryCell)));
	cudaDeviceSynchronize();
	cudaMemcpy(memoryCell, memory, sizeof(MemoryCell), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double *cdw, *idp, *fdp, *cdp;
	cdw = (double *)malloc(sizeof(double) * memoryCell->nConnections);
	idp = (double *)malloc(sizeof(double) * memoryCell->nConnections);
	fdp = (double *)malloc(sizeof(double) * memoryCell->nConnections);
	cdp = (double *)malloc(sizeof(double) * memoryCell->nConnections);

	cudaMemcpy(cdw, memoryCell->cell_data_weight, (sizeof(double) * memoryCell->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(idp, memoryCell->input_data_partial, (sizeof(double) * memoryCell->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(fdp, memoryCell->forget_data_partial, (sizeof(double) * memoryCell->nConnections), cudaMemcpyDeviceToHost);
	cudaMemcpy(cdp, memoryCell->cell_data_partial, (sizeof(double) * memoryCell->nConnections), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	memcpy(&(memoryCell->cell_data_weight), &cdw, (sizeof(double *)));
	memcpy(&(memoryCell->input_data_partial), &idp, (sizeof(double *)));
	memcpy(&(memoryCell->forget_data_partial), &fdp, (sizeof(double *)));
	memcpy(&(memoryCell->cell_data_partial), &cdp, (sizeof(double *)));

	return memoryCell;
}


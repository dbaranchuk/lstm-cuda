#include "Neuron.cuh"

long long Neuron::n = 0;

Neuron::Neuron(int size) {
	activation = 0; activationPrime = 0;
	connections = size;
	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);
	weightedError = (double *)malloc(sizeof(double) * size);
	weight = (double *)malloc(sizeof(double) * size);
	impulse = (double *)calloc(size, sizeof(double));
	for (int i = 0; i < size; i++) {
		weight[i] = (d(g));
	}
}

Neuron::~Neuron() {}

__device__ double Neuron::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

__device__ double Neuron::sigmoidPrime(double input) {
	return sigmoid(input) * (1 - sigmoid(input));
}

__device__ double Neuron::activate(double input) {
	return tanh(input);
}

__device__ double Neuron::activatePrime(double input) {
	return (1 - (tanh(input) * tanh(input)));
}

__device__ double Neuron::forward(double *input) {
	double sum = 0;
	// find the weighted sum of all input
	for (int i = 0; i < connections; i++) {
		sum += input[i] * weight[i];
	}
	activation = activate(sum);
	activationPrime = activatePrime(sum);
	return activation;
}

__device__ double *Neuron::backward(double errorPrime, double learningRate) {
	// update all weights
	for (int i = 0; i < connections; i++) {
		weightedError[i] = (errorPrime * weight[i] * activationPrime);
		weight[i] -= learningRate * errorPrime * impulse[i];
	}
	return weightedError;
}

Neuron *Neuron::copyToGPU(Neuron *data) {
	Neuron *neuron;
	cudaMalloc((void **)&neuron, (sizeof(Neuron)));
	cudaDeviceSynchronize();
	cudaMemcpy(neuron, data, sizeof(Neuron), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	double *e;
	double *w;
	double *i;
	cudaMalloc((void **)&e, (sizeof(double) * data->connections));
	cudaMalloc((void **)&w, (sizeof(double) * data->connections));
	cudaMalloc((void **)&i, (sizeof(double) * data->connections));
	cudaDeviceSynchronize();

	cudaMemcpy(e, data->weightedError, (sizeof(double) * data->connections), cudaMemcpyHostToDevice);
	cudaMemcpy(w, data->weight, (sizeof(double) * data->connections), cudaMemcpyHostToDevice);
	cudaMemcpy(i, data->impulse, (sizeof(double) * data->connections), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(&(neuron->weightedError), &e, sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(neuron->weight), &w, sizeof(double *), cudaMemcpyHostToDevice);
	cudaMemcpy(&(neuron->impulse), &i, sizeof(double *), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	return neuron;
}

Neuron *Neuron::copyFromGPU(Neuron *data) {
	Neuron *neuron;
	neuron = (Neuron *)malloc((sizeof(Neuron)));
	cudaDeviceSynchronize();
	cudaMemcpy(neuron, data, sizeof(Neuron), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double *e;
	double *w;
	double *i;
	e = (double *)malloc(sizeof(double) * neuron->connections);
	w = (double *)malloc(sizeof(double) * neuron->connections);
	i = (double *)malloc(sizeof(double) * neuron->connections);
	cudaDeviceSynchronize();

	cudaMemcpy(e, neuron->weightedError, (sizeof(double) * neuron->connections), cudaMemcpyDeviceToHost);
	cudaMemcpy(w, neuron->weight, (sizeof(double) * neuron->connections), cudaMemcpyDeviceToHost);
	cudaMemcpy(i, neuron->impulse, (sizeof(double) * neuron->connections), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	memcpy(&(neuron->weightedError), &e, sizeof(double *));
	memcpy(&(neuron->weight), &w, sizeof(double *));
	memcpy(&(neuron->impulse), &i, sizeof(double *));
	cudaDeviceSynchronize();

	return neuron;
}


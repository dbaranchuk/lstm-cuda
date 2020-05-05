#include "LSTMNetwork.cuh"

__global__ void forwardPass(Neuron **neurons, double *connections, double *activations, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			activations[idx] = neurons[idx]->forward(connections);
		}
	}
}

__global__ void backwardPass(Neuron **neurons, double *weightedError, double *errorSum, double learningRate, int connections, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			double *contribution = neurons[idx]->backward(weightedError[idx], learningRate);
			for (int j = 0; j < connections; j++) {
				errorSum[j] += contribution[j];
			}
		}
	}
}

__global__ void forwardPassLSTM(MemoryBlock **blocks, double *connections, double *activations, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			double *blockActivation = blocks[idx]->forward(connections);
			for (int j = 0; j < blocks[i]->nCells; j++) activations[idx * blocks[i]->nCells + j] = blockActivation[j];
		}
	}
}

__global__ void backwardPassLSTM(MemoryBlock **blocks, double **weightedError, double *errorSum, double learningRate, int connections, int size, int cycles) {
	int maxId = gridDim.x * blockDim.x;
	for (int i = 0; i < (cycles); i++) {
		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
		if (idx < size) {
			double *contribution = blocks[idx]->backward(weightedError[idx], learningRate);
			for (int j = 0; j < connections; j++) {
				errorSum[j] += contribution[j];
			}
		}
	}
}

LSTMNetwork::LSTMNetwork(int is, int b, int c, double lr, int num_classes) {
	inputSize = is;
	learningRate = lr;
	for (int i = 0; i < b; i++) {
		blocks.push_back(MemoryBlock(c, is));
	}
	for (int i = 0; i < num_classes; i++)
		layer.push_back(Neuron(b * c));
}

LSTMNetwork::~LSTMNetwork() {}

vector<double> LSTMNetwork::classify(vector<double> input) {
	double *output = (double *)malloc(sizeof(double) * blocks.size() * blocks[0].nCells),
			*connections;
	cudaMalloc((void **)&connections, sizeof(double) * input.size());
	cudaMemcpy(&connections[0], &input[0], (sizeof(double) * input.size()), cudaMemcpyHostToDevice);
	if (input.size() == inputSize) {
		// calculate activations from bottom up
		double *activations;
		cudaMalloc((void **)&activations, (sizeof(double) * blocks.size() * blocks[0].nCells));

		MemoryBlock **deviceBlocks, **blockBuffer = (MemoryBlock **)malloc(sizeof(MemoryBlock *) * blocks.size());
		for (int i = 0; i < blocks.size(); i++) {
			cudaMemcpy(&(blocks[i].impulse[0]), &connections[0], (sizeof(double) * blocks[i].nConnections), cudaMemcpyDeviceToHost);
		}
		cudaMalloc((void **)&deviceBlocks, sizeof(MemoryBlock *) * blocks.size());
		for (int i = 0; i < blocks.size(); i++) {
			MemoryBlock *db = MemoryBlock::copyToGPU(&blocks[i]);
			cudaMemcpy(&deviceBlocks[i], &db, sizeof(MemoryBlock *), cudaMemcpyHostToDevice);
		} forwardPassLSTM<<<maxBlocks, maxThreads>>>(deviceBlocks, connections, activations, blocks.size(),
		                                             ceil((double)blocks.size() / (double)(maxBlocks * maxThreads)));
		cudaDeviceSynchronize();

		cudaMemcpy(&blockBuffer[0], &deviceBlocks[0], (sizeof(MemoryBlock *) * blocks.size()), cudaMemcpyDeviceToHost);
		for (int i = 0; i < blocks.size(); i++) {
			blocks[i] = *MemoryBlock::copyFromGPU(blockBuffer[i]);
		} free(blockBuffer);
		cudaFree(deviceBlocks);

		cudaFree(connections);
		cudaMalloc((void **)&connections, (sizeof(double) * blocks.size() * blocks[0].nCells));
		cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * blocks.size() * blocks[0].nCells), cudaMemcpyDeviceToDevice);
		cudaFree(activations);
		free(output);
		output = (double *)malloc(sizeof(double) * layer.size());

        // Layer
		cudaMalloc((void **)&activations, (sizeof(double) * layer.size()));

		Neuron **deviceNeurons, **neuronBuffer = (Neuron **)malloc(sizeof(Neuron *) * layer.size());
		for (int j = 0; j < layer.size(); j++) {
			cudaMemcpy(&(layer[j].impulse[0]), &connections[0], (sizeof(double) * layer[j].connections), cudaMemcpyDeviceToHost);
		}
		cudaMalloc((void **)&deviceNeurons, sizeof(Neuron *) * layer.size());
		for (int j = 0; j < layer.size(); j++) {
			Neuron *dn = Neuron::copyToGPU(&layer[j]);
			cudaMemcpy(&deviceNeurons[j], &dn, sizeof(Neuron *), cudaMemcpyHostToDevice);
		} forwardPass<<<maxBlocks, maxThreads>>>(deviceNeurons, connections, activations, layer.size(), ceil((double)layer.size() / (double)(maxBlocks * maxThreads)));
		cudaDeviceSynchronize();

		cudaFree(connections);
		cudaMalloc((void **)&connections, (sizeof(double) * layer.size()));
		cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * layer.size()), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&neuronBuffer[0], &deviceNeurons[0], (sizeof(Neuron *) * layer.size()), cudaMemcpyDeviceToHost);
		for (int j = 0; j < layer.size(); j++) {
			layer[j] = *Neuron::copyFromGPU(neuronBuffer[j]);
		} 
		cudaMemcpy(&output[0], &activations[0], (sizeof(double) * layer.size()), cudaMemcpyDeviceToHost);
		cudaFree(activations);
		cudaFree(deviceNeurons);
		free(neuronBuffer);
		
		vector<double> result(&output[0], &output[layer.size()]);
		free(output);
		cudaFree(connections);
		return result;
	} else return vector<double>();
}

vector<double> LSTMNetwork::train(vector<double> input, vector<double> target) {
	double *output = (double *)malloc(blocks.size() * blocks[0].nCells * sizeof(double)),
			*connections;
	cudaMalloc((void **)&connections, sizeof(double) * input.size());
	cudaMemcpy(&connections[0], &input[0], (sizeof(double) * input.size()), cudaMemcpyHostToDevice);
	if (input.size() != inputSize) {
	    cout << "Target size mismatch" << endl;
		return vector<double>();
	}
    // start forward pass
    double *activations;
    cudaMalloc((void **)&activations, (sizeof(double) * blocks.size() * blocks[0].nCells));
    MemoryBlock **deviceBlocks;
    for (int i = 0; i < blocks.size(); i++) {
        cudaMemcpy(&(blocks[i].impulse[0]), &connections[0], (sizeof(double) * blocks[i].nConnections), cudaMemcpyDeviceToHost);
    } cudaMalloc((void **)&deviceBlocks, sizeof(MemoryBlock *) * blocks.size());
    for (int i = 0; i < blocks.size(); i++) {
        MemoryBlock *db = MemoryBlock::copyToGPU(&blocks[i]);
        cudaMemcpy(&deviceBlocks[i], &db, sizeof(MemoryBlock *), cudaMemcpyHostToDevice);
    } forwardPassLSTM<<<maxBlocks, maxThreads>>>(deviceBlocks, connections, activations, blocks.size(), ceil((double)blocks.size() / (double)(maxBlocks * maxThreads)));
    cudaDeviceSynchronize();
    cudaFree(connections);
    cudaMalloc((void **)&connections, (sizeof(double) * blocks.size() * blocks[0].nCells));
    cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * blocks.size() * blocks[0].nCells), cudaMemcpyDeviceToDevice);
    cudaFree(activations);
    free(output);

    cout << blocks.size() * blocks[0].nCells;
    for (int i = 0; i < blocks.size(); i++)

        cout << activations[i];

    output = (double *)malloc(sizeof(double) * layer.size());

    cudaMalloc((void **)&activations, (sizeof(double) * layer.size()));

    Neuron **layerNeurons;
    for (int j = 0; j < layer.size(); j++) {
        cudaMemcpy(&(layer[j].impulse[0]), &connections[0], (sizeof(double) * layer[j].connections), cudaMemcpyDeviceToHost);
    }
    cudaMalloc((void **)&layerNeurons, sizeof(Neuron *) * layer.size());
    for (int j = 0; j < layer.size(); j++) {
        Neuron *dn = Neuron::copyToGPU(&layer[j]);
        cudaMemcpy(&layerNeurons[j], &dn, sizeof(Neuron *), cudaMemcpyHostToDevice);
    }
    forwardPass<<<maxBlocks, maxThreads>>>(layerNeurons, connections, activations, layer.size(), ceil((double)layer.size() / (double)(maxBlocks * maxThreads)));
    cudaDeviceSynchronize();
    cudaFree(connections);
    cudaMalloc((void **)&connections, (sizeof(double) * layer.size()));

    cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * layer.size()), cudaMemcpyDeviceToDevice);
    cudaFree(activations);
    cudaFree(connections);



    // start backward pass
    double *weightedError;
    cudaMalloc((void **)&weightedError, (sizeof(double) * layer.size()));
    for (int i = 0; i < layer.size(); i++) {
        double error = (output[i] - target[i]);
        output[i] = error;
        cudaMemcpy(&weightedError[i], &error, sizeof(double), cudaMemcpyHostToDevice);

    }
    double *errorSum;
    cudaMalloc((void **)&errorSum, (sizeof(double) * layer[0].connections));
    cudaMemset(&errorSum[0], 0, (sizeof(double) * layer[0].connections));

    // compute the gradient
    backwardPass<<<maxBlocks, maxThreads>>>(layerNeurons, weightedError, errorSum, learningRate, layer[0].connections, layer.size(), ceil((double)layer.size() / (double)(maxBlocks * maxThreads)));
    cudaDeviceSynchronize();
    cudaFree(weightedError);
    cudaMalloc((void **)&weightedError, (sizeof(double) * layer[0].connections));
    cudaMemcpy(&weightedError[0], &errorSum[0], (sizeof(double) * layer[0].connections), cudaMemcpyDeviceToDevice);

    Neuron **neuronBuffer = (Neuron **)malloc(sizeof(Neuron) * layer.size());
    cudaMemcpy(&neuronBuffer[0], &layerNeurons, (sizeof(Neuron *) * layer.size()), cudaMemcpyDeviceToHost);
    for (int j = 0; j < layer.size(); j++) {
        layer[j] = *Neuron::copyFromGPU(neuronBuffer[j]);
    } free(neuronBuffer);
    cudaFree(layerNeurons);
    cudaFree(errorSum);


    double **errorChunks;
    cudaMalloc((void **)&errorChunks, (sizeof(double *) * blocks.size()));
    cudaMalloc((void **)&errorSum, (sizeof(double) * blocks[0].nConnections));
    cudaMemset(&errorSum[0], 0.0, (sizeof(double) * blocks[0].nConnections));
    for (int i = 0; i < (blocks.size()); i++) {
        double *chunk;
        cudaMalloc((void **)&chunk, (sizeof(double) * blocks[i].nCells));
        cudaMemcpy(&chunk[0], &weightedError[(i * blocks[i].nCells)], (sizeof(double) * blocks[i].nCells), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&errorChunks[i], &chunk, (sizeof(double *)), cudaMemcpyHostToDevice);
    } backwardPassLSTM<<<maxBlocks, maxThreads>>>(deviceBlocks, errorChunks, errorSum, learningRate, blocks[0].nConnections, blocks.size(), ceil((double)blocks.size() / (double)(maxBlocks * maxThreads)));
    cudaDeviceSynchronize();

    MemoryBlock **blockBuffer = (MemoryBlock **)malloc(sizeof(MemoryBlock *) * blocks.size());
    //cout << blocks.size() << " copy blocks " <<
    cudaMemcpy(blockBuffer, deviceBlocks, (sizeof(MemoryBlock *) * blocks.size()), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocks.size(); i++) {
        MemoryBlock temp = *MemoryBlock::copyFromGPU(blockBuffer[i]);
        blocks[i] = temp;
    }

    cudaFree(deviceBlocks);
    cudaFree(weightedError);
    cudaFree(errorChunks);
    cudaFree(errorSum);

    vector<double> result(&output[0], &output[layer.size()]);
    free(output);
    return result;
}

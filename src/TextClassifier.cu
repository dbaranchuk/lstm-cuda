#include "TextClassifier.cuh"

__global__ void forwardPass(Neuron **neurons, double *connections, double *activations, int size) {
	//int maxId = gridDim.x * blockDim.x;
	//int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
	//if (idx < size) {
	for (int i = 0; i < size; i++)
		activations[i] = neurons[i]->forward(connections);
	//}
}

//__global__ void backwardPass(Neuron **neurons, double *weightedError, double *errorSum,
//							 double learningRate, int connections, int size) {
//	int maxId = gridDim.x * blockDim.x;
//	int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
//	if (idx < size) {
//		double *contribution = neurons[idx]->backward(weightedError[idx], learningRate);
//		for (int j = 0; j < connections; j++) {
//			errorSum[j] += contribution[j];
//		}
//	}
//}

//__global__ void forwardPassLSTM(MemoryBlock **blocks, double *connections, double *activations, int size, int cycles) {
//	int maxId = gridDim.x * blockDim.x;
//	for (int i = 0; i < (cycles); i++) {
//		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
//		if (idx < size) {
//			double *blockActivation = blocks[idx]->forward(connections);
//			for (int j = 0; j < blocks[i]->nCells; j++) activations[idx * blocks[i]->nCells + j] = blockActivation[j];
//		}
//	}
//}

__global__ void forwardPassLSTM(MemoryBlock *block, double **connections, double *activations, int cycles) {
    double *local_activations;
    for (int i = 0; i < cycles; i++) {
		local_activations = block->forward(connections[i]);
	}
    for (int i = 0; i < block->nCells; i++)
        activations[i] = local_activations[i];
}

//__global__ void backwardPassLSTM(MemoryBlock **blocks, double **weightedError, double *errorSum, double learningRate, int connections, int size, int cycles) {
//	int maxId = gridDim.x * blockDim.x;
//	for (int i = 0; i < (cycles); i++) {
//		int idx = (threadIdx.x + blockIdx.x * blockDim.x) + (maxId * i);
//		if (idx < size) {
//			double *contribution = blocks[idx]->backward(weightedError[idx], learningRate);
//			for (int j = 0; j < connections; j++) {
//				errorSum[j] += contribution[j];
//			}
//		}
//	}
//}

TextClassifier::TextClassifier(int is, int c, double lr, int num_classes) {
	inputSize = is;
	learningRate = lr;
	block = new MemoryBlock(c, is);

	layer = vector<Neuron>;
	for (int i = 0; i < num_classes; i++)
		layer.push_back(Neuron(c));
}

TextClassifier::~TextClassifier() {}

//vector<double> TextClassifier::classify(vector<double> input) {
//	double *connections;
//	cudaMalloc((void **)&connections, sizeof(double) * input.size());
//	cudaMemcpy(&connections[0], &input[0], (sizeof(double) * input.size()), cudaMemcpyHostToDevice);
//	if (input.size() == inputSize) {
//		// calculate activations from bottom up
//		double *activations;
//		cudaMalloc((void **)&activations, (sizeof(double) * blocks.size() * blocks[0].nCells));
//
//		MemoryBlock **deviceBlocks, **blockBuffer = (MemoryBlock **)malloc(sizeof(MemoryBlock *) * blocks.size());
//		for (int i = 0; i < blocks.size(); i++) {
//			cudaMemcpy(&(blocks[i].impulse[0]), &connections[0], (sizeof(double) * blocks[i].nConnections), cudaMemcpyDeviceToHost);
//		}
//		cudaMalloc((void **)&deviceBlocks, sizeof(MemoryBlock *) * blocks.size());
//		for (int i = 0; i < blocks.size(); i++) {
//			MemoryBlock *db = MemoryBlock::copyToGPU(&blocks[i]);
//			cudaMemcpy(&deviceBlocks[i], &db, sizeof(MemoryBlock *), cudaMemcpyHostToDevice);
//		} forwardPassLSTM<<<maxBlocks, maxThreads>>>(deviceBlocks, connections, activations, blocks.size(),
//		                                             ceil((double)blocks.size() / (double)(maxBlocks * maxThreads)));
//		cudaDeviceSynchronize();
//
//		cudaMemcpy(&blockBuffer[0], &deviceBlocks[0], (sizeof(MemoryBlock *) * blocks.size()), cudaMemcpyDeviceToHost);
//		for (int i = 0; i < blocks.size(); i++) {
//			blocks[i] = *MemoryBlock::copyFromGPU(blockBuffer[i]);
//		} free(blockBuffer);
//		cudaFree(deviceBlocks);
//
//		cudaFree(connections);
//		cudaMalloc((void **)&connections, (sizeof(double) * blocks.size() * blocks[0].nCells));
//		cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * blocks.size() * blocks[0].nCells), cudaMemcpyDeviceToDevice);
//		cudaFree(activations);
//
//        // logits_layer
//		cudaMalloc((void **)&activations, (sizeof(double) * logits_layer.size()));
//
//		Neuron **deviceNeurons, **neuronBuffer = (Neuron **)malloc(sizeof(Neuron *) * logits_layer.size());
//		for (int j = 0; j < logits_layer.size(); j++) {
//			cudaMemcpy(&(layer[j].impulse[0]), &connections[0], (sizeof(double) * logits_layer[j].connections), cudaMemcpyDeviceToHost);
//		}
//		cudaMalloc((void **)&deviceNeurons, sizeof(Neuron *) * logits_layer.size());
//		for (int j = 0; j < logits_layer.size(); j++) {
//			Neuron *dn = Neuron::copyToGPU(&layer[j]);
//			cudaMemcpy(&deviceNeurons[j], &dn, sizeof(Neuron *), cudaMemcpyHostToDevice);
//		} forwardPass<<<maxBlocks, maxThreads>>>(deviceNeurons, connections, activations, logits_layer.size(), ceil((double)layer.size() / (double)(maxBlocks * maxThreads)));
//		cudaDeviceSynchronize();
//
//		cudaFree(connections);
//		cudaMalloc((void **)&connections, (sizeof(double) * logits_layer.size()));
//		cudaMemcpy(&connections[0], &activations[0], (sizeof(double) * logits_layer.size()), cudaMemcpyDeviceToDevice);
//		cudaMemcpy(&neuronBuffer[0], &deviceNeurons[0], (sizeof(Neuron *) * logits_layer.size()), cudaMemcpyDeviceToHost);
//		for (int j = 0; j < logits_layer.size(); j++) {
//			layer[j] = *Neuron::copyFromGPU(neuronBuffer[j]);
//		}
//		double *output = (double *)malloc(sizeof(double) * logits_layer.size());
//		cudaMemcpy(&output[0], &activations[0], (sizeof(double) * logits_layer.size()), cudaMemcpyDeviceToHost);
//		cudaFree(activations);
//		cudaFree(deviceNeurons);
//		free(neuronBuffer);
//
//		vector<double> result(&output[0], &output[layer.size()]);
//		free(output);
//		cudaFree(connections);
//		return result;
//	} else return vector<double>();
//}

double TextClassifier::train(vector<vector<double>> &inputs, vector<double> &target) {
	if (inputs[0].size() != inputSize) {
	    cout << "Target size mismatch" << endl;
		return 0.0;
	}
    // Load input data to GPU
    double **connections;
	double *lstm_activations;
	cudaMalloc((void **)&connections, sizeof(double *) * inputs.size());
    cudaMalloc((void **)&lstm_activations, sizeof(double) * block.nCells);
    for (int i = 0; i < inputs.size(); i++) {
        cudaMalloc((void **) &connections[i], sizeof(double) * inputs[0].size());
        cudaMemcpy(connections[i].data(), inputs[i].data(),
                   sizeof(double) * inputs[0].size(), cudaMemcpyHostToDevice);
    }
	cout << inputs[0].size() << " " << block.nConnections;
	// TODO
	for (int i = 0; i < inputs.size(); i++) {
        cudaMemcpy(block.impulses[i].data(), connections[i].data(),
                   (sizeof(double) * block.nConnections), cudaMemcpyDeviceToHost);
    }
    MemoryBlock *device_block = MemoryBlock::copyToGPU(block);
    forwardPassLSTM<<<maxBlocks, maxThreads>>>(device_block, connections, lstm_activations, inputs.size());
    cudaDeviceSynchronize();
	cudaFree(connections);

    // lstm_activations become new connections for logit logits_layer

	double *logits_activations;
	cudaMalloc((void **)&logits_activations, sizeof(double) * logits_layer.size());

	// Logits

	// Put lstm activation to impulse for backprop
    Neuron **layerNeurons;
    for (int j = 0; j < logits_layer.size(); j++) {
        cudaMemcpy(logits_layer[j].impulse.data(), lstm_activations.data(),
        		sizeof(double) * logits_layer[j].connections, cudaMemcpyDeviceToHost);
    }
	// Copy linear logits_layer to device
    cudaMalloc((void **)&layerNeurons, sizeof(Neuron *) * logits_layer.size());
    for (int j = 0; j < logits_layer.size(); j++) {
        Neuron *device_neuron = Neuron::copyToGPU(&logits_layer[j]);
        cudaMemcpy(&layerNeurons[j], &device_neuron, sizeof(Neuron *), cudaMemcpyHostToDevice);
    }
    
    // Logits forward
    forwardPass<<<maxBlocks, maxThreads>>>(layerNeurons, lstm_activations, 
    		                               logits_activations, logits_layer.size());
    cudaDeviceSynchronize();
    cudaFree(lstm_activations);
    
    double *output = (double *)malloc(sizeof(double) * logits_layer.size());
    cudaMemcpy(output.data(), logits_activations.data(), 
    		   sizeof(double) * logits_layer.size(), cudaMemcpyDeviceToHost);

    cudaFree(logits_activations);

    cout << logits_layer.size() << "\n";
	double loss = 0.0;
    for (int i = 0; i < logits_layer.size(); i++)
        loss += output[i];

    return loss
    ///////////////////////////////////////////////////////////////

    // start backward pass
//    double *weightedError;
//    cudaMalloc((void **)&weightedError, (sizeof(double) * logits_layer.size()));
//    for (int i = 0; i < logits_layer.size(); i++) {
//        double error = (output[i] - target[i]);
//        output[i] = error;
//        cudaMemcpy(&weightedError[i], &error, sizeof(double), cudaMemcpyHostToDevice);
//
//    }
//    double *errorSum;
//    cudaMalloc((void **)&errorSum, (sizeof(double) * logits_layer[0].connections));
//    cudaMemset(&errorSum[0], 0, (sizeof(double) * logits_layer[0].connections));
//
//    // compute the gradient
//    backwardPass<<<maxBlocks, maxThreads>>>(layerNeurons, weightedError, errorSum, learningRate, logits_layer[0].connections, logits_layer.size(), ceil((double)layer.size() / (double)(maxBlocks * maxThreads)));
//    cudaDeviceSynchronize();
//    cudaFree(weightedError);
//    cudaMalloc((void **)&weightedError, (sizeof(double) * logits_layer[0].connections));
//    cudaMemcpy(&weightedError[0], &errorSum[0], (sizeof(double) * logits_layer[0].connections), cudaMemcpyDeviceToDevice);
//
//    Neuron **neuronBuffer = (Neuron **)malloc(sizeof(Neuron) * logits_layer.size());
//    cudaMemcpy(&neuronBuffer[0], &layerNeurons, (sizeof(Neuron *) * logits_layer.size()), cudaMemcpyDeviceToHost);
//    for (int j = 0; j < logits_layer.size(); j++) {
//        logits_layer[j] = *Neuron::copyFromGPU(neuronBuffer[j]);
//    } free(neuronBuffer);
//    cudaFree(layerNeurons);
//    cudaFree(errorSum);
//
//
//    double **errorChunks;
//    cudaMalloc((void **)&errorChunks, (sizeof(double *) * blocks.size()));
//    cudaMalloc((void **)&errorSum, (sizeof(double) * blocks[0].nConnections));
//    cudaMemset(&errorSum[0], 0.0, (sizeof(double) * blocks[0].nConnections));
//    for (int i = 0; i < (blocks.size()); i++) {
//        double *chunk;
//        cudaMalloc((void **)&chunk, (sizeof(double) * blocks[i].nCells));
//        cudaMemcpy(&chunk[0], &weightedError[(i * blocks[i].nCells)], (sizeof(double) * blocks[i].nCells), cudaMemcpyDeviceToDevice);
//        cudaMemcpy(&errorChunks[i], &chunk, (sizeof(double *)), cudaMemcpyHostToDevice);
//    } backwardPassLSTM<<<maxBlocks, maxThreads>>>(deviceBlocks, errorChunks, errorSum, learningRate, blocks[0].nConnections, blocks.size(), ceil((double)blocks.size() / (double)(maxBlocks * maxThreads)));
//    cudaDeviceSynchronize();
//
//    MemoryBlock **blockBuffer = (MemoryBlock **)malloc(sizeof(MemoryBlock *) * blocks.size());
//    //cout << blocks.size() << " copy blocks " <<
//    cudaMemcpy(blockBuffer, deviceBlocks, (sizeof(MemoryBlock *) * blocks.size()), cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i < blocks.size(); i++) {
//        MemoryBlock temp = *MemoryBlock::copyFromGPU(blockBuffer[i]);
//        blocks[i] = temp;
//    }
//
//    cudaFree(deviceBlocks);
//    cudaFree(weightedError);
//    cudaFree(errorChunks);
//    cudaFree(errorSum);

//    vector<double> result(&output[0], &output[layer.size()]);
//    free(output);
//    return result;
}

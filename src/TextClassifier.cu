#include "TextClassifier.cuh"


__global__ void logits_forward_pass(Neuron **neurons, double *connections, double *activations, int size) {
    //int maxId = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        activations[idx] = neurons[idx]->forward(connections);
    }
}

__global__ void lstm_forward_pass(LSTMCell *block, double *connections, double *activations, int size)
{
    double cellSum[100]; //TODO
    double inputSum = block->bias[0];
    double forgetSum = block->bias[1];
    double outputSum = block->bias[2];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < block->nConnections) {
        for (int j = 0; j < block->nCells; j++) {
            cellSum[j] += connections[idx] * block->cells[j]->cell_data_weight[idx];
        }
        inputSum += connections[idx] * block->input_data_weight[idx];
        forgetSum += connections[idx] * block->forget_data_weight[idx];
        outputSum += connections[idx] * block->output_data_weight[idx];
    }

    if (idx < block->nCells) {
        inputSum += block->input_hidden_weight[idx] * block->cells[idx]->feedback;
        forgetSum += block->forget_hidden_weight[idx] * block->cells[idx]->feedback;
        outputSum += block->output_hidden_weight[idx] * block->cells[idx]->feedback;

        block->cells[idx]->previousState = block->cells[idx]->state;
        block->cells[idx]->state *= block->forgetGate(forgetSum);
        block->cells[idx]->state += block->cells[idx]->activateIn(cellSum[idx]) * block->inputGate(inputSum);

        // compute output of memory cell
        block->cells[idx]->previousFeedback = block->cells[idx]->feedback;
        block->cells[idx]->feedback = block->cells[idx]->activateOut(block->cells[idx]->state) * block->outputGate(outputSum);
        activations[idx] = block->cells[idx]->feedback;
    }
}


TextClassifier::TextClassifier(int input_size, int hidden_size, double lr, int num_classes) {
	inputSize = input_size;
	learningRate = lr;
	block = new LSTMCell(hidden_size, input_size);

	for (int i = 0; i < num_classes; i++)
        logits_layer.push_back(Neuron(hidden_size));
}

TextClassifier::~TextClassifier() {}


double TextClassifier::train(vector<double> &inputs, vector<double> &target) {
    // Load input data to GPU
    double *connections, *lstm_activations;
    cudaMalloc((void **) &connections, sizeof(double) * inputs.size());
    cudaMalloc((void **) &lstm_activations, sizeof(double) * block->nCells);
    cudaMemcpy(connections, inputs.data(),
               sizeof(double) * inputs.size(), cudaMemcpyHostToDevice);

    LSTMCell *device_block = LSTMCell::copyToGPU(block);
    for (int i = 0; i < 100; i++) {
        lstm_forward_pass<<< maxBlocks, maxThreads >>>(device_block, connections + block->nConnections * i,
                                                       lstm_activations, block->nConnections);
    }
    cudaDeviceSynchronize();
    cudaFree(connections);

    // lstm_activations become new connections for logit logits_layer

    double *logits_activations;
    cudaMalloc((void **) &logits_activations, sizeof(double) * logits_layer.size());

    // Logits

    // Put lstm activation to impulse for backprop
    Neuron **layerNeurons;
    for (int j = 0; j < logits_layer.size(); j++) {
        cudaMemcpy(&logits_layer[j].impulse[0], &lstm_activations[0],
                   sizeof(double) * logits_layer[j].connections, cudaMemcpyDeviceToHost);
    }
    // Copy linear logits_layer to device
    cudaMalloc((void **) &layerNeurons, sizeof(Neuron *) * logits_layer.size());
    for (int j = 0; j < logits_layer.size(); j++) {
        Neuron *device_neuron = Neuron::copyToGPU(&logits_layer[j]);
        cudaMemcpy(&layerNeurons[j], &device_neuron, sizeof(Neuron *), cudaMemcpyHostToDevice);
    }

    // Logits forward
    logits_forward_pass <<< maxBlocks, maxThreads >>> (layerNeurons, lstm_activations,
            logits_activations, logits_layer.size());
    cudaDeviceSynchronize();
    cudaFree(lstm_activations);

    double *output = (double *) malloc(sizeof(double) * logits_layer.size());
    cudaMemcpy(&output[0], &logits_activations[0],
               sizeof(double) * logits_layer.size(), cudaMemcpyDeviceToHost);

    cudaFree(logits_activations);
    double loss = 0.0;
    for (int i = 0; i < logits_layer.size(); i++)
        loss += output[i];

    return loss;
}
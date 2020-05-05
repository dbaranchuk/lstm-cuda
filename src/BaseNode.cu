#include "BaseNode.cuh"

BaseNode::BaseNode() {
}

BaseNode::~BaseNode() {
}

__device__ double BaseNode::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

__device__ double BaseNode::sigmoidPrime(double input) {
	return sigmoid(input) * (1 - sigmoid(input));
}

__device__ double BaseNode::activationFunction(double input) {
	return tanh(input);
}

__device__ double BaseNode::activationFunctionPrime(double input) {
	return (1 - (tanh(input) * tanh(input)));
}


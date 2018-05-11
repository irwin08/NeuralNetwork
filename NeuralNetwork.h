#include <Eigen/Dense>
#include <iostream>
#include <vector>

using namespace Eigen;

class NeuralNetwork
{
	public:
		// Takes in a vector which gives the number of neurons corresponding to a layer.
		// For example, a neural net with an input layer with n=3 neurons, a hidden layer with n=5 neurons and an output layer with n=3 neurons
		// will take the vector [3, 5, 3].
		NeuralNetwork(std::vector<int> layerSizes);
		// assigns values to weights and biases by first looking for an already existing NN, then defaulting to random values if no NN is found.
		void initializeNeuralNet();
		
		// returns the index number corresponding to the neuron activated in the output layer (starts at i=0)
		// NOTE: This can be generalized by returning a vector with an arbitrary number of neurons activated over some threshold instead.
		int getOutput(VectorXf a);
		
		// trains neural net using SGD algorithm - takes in a set of input layers to train and a set of desired outputs for said layers.
		void stochasticGradientDescent(std::vector<VectorXf> trainingSet, std::vector<VectorXf> trainingAnswer, float learningRate);
		
		
		
	private:
		
		std::vector<int> _layerSizes; 
		std::vector<MatrixXf> _weights;  		// weights is a vector, where each element corresponds to the relevent weight matrix for the layer.
		std::vector<VectorXf> _activations; 	// vector in which each element corresponds to the activation vector for a layer.
		std::vector<VectorXf> _biases; 			// vector in which each element corresponds to the bias vector for a layer.
		
		VectorXf sigmoid(VectorXf z);
		VectorXf sigmoidPrime(VectorXf z);
	
};
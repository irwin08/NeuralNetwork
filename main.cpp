#include <iostream>
#include "NeuralNetwork.h"

using namespace std;

int main()
{
	vector<int> nnSize;
	nnSize.push_back(6);
	nnSize.push_back(8);
	nnSize.push_back(6);
	
	NeuralNetwork *nn = new NeuralNetwork(nnSize);
	
	VectorXf a(6);
	a(0) = 0.0;
	a(1) = 0.0;
	a(2) = 0.0;
	a(3) = 0.9;
	a(4) = 0.0;
	a(5) = 0.0;
	
	VectorXf b(6);
	b(0) = 0.0;
	b(1) = 0.0;
	b(2) = 1.0;
	b(3) = 0.0;
	b(4) = 0.0;
	b(5) = 0.0;
	
	vector<VectorXf> trainingSet;
	trainingSet.push_back(a);
	
	vector<VectorXf> trainingAnswers;
	trainingAnswers.push_back(b);
	
	cout << "Starting Neural Network..." << endl;
	
	cout << nn->getOutput(a)<< endl;
	
	cout << "Beginning training..." << endl;
	
	nn->stochasticGradientDescent(trainingSet, trainingAnswers, 0.1);
	
	cout << "Training complete." << endl;
	
	cout << nn->getOutput(a)<< endl;
	
	nn->saveNeuralNetwork();
	
	return 0;
}
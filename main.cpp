#include <iostream>
#include "NeuralNetwork.h"

using namespace std;

int main()
{
	std:vector<int> nnSize;
	nnSize.push_back(6);
	nnSize.push_back(8);
	nnSize.push_back(6);
	
	NeuralNetwork *nn = new NeuralNetwork(nnSize);
	
	VectorXf a(6);
	a(0) = 1;
	a(1) = 2;
	a(2) = 3;
	a(3) = 4;
	a(4) = 5;
	a(5) = 6;
	
	cout << nn->getOutput(a)<< endl;
	return 0;
}
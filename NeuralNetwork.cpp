#include "NeuralNetwork.h"
#include <cmath>
#include <string>
#include <fstream>
#include <random>

NeuralNetwork::NeuralNetwork(std::vector<int> layerSizes)
{
	_layerSizes = layerSizes;
	
	//this initialization is probably redundant, but I'd rather be on the safe side for now -- I don't want to blow up my computer.
	_weights.push_back(MatrixXf::Zero(1,1));	//there are no weights for the base case, so set to zero of arbitrary size.
	_biases.push_back(VectorXf::Zero(1)); 		// there are no biases for the base case, so set to zero of arbitrary size.
	for(int i = 1; i < _layerSizes.size(); i++)
	{
		_weights.push_back(MatrixXf::Zero(_layerSizes[i], _layerSizes[i-1]));	// initializes weight matrix for each layer where r=oldLayer c=newLayer
		_biases.push_back(VectorXf::Zero(_layerSizes[i]));						// initializes bias vector for each layer.
	}
	
	// this function name is awesome.
	initializeNeuralNet();
	
}


void NeuralNetwork::initializeNeuralNet()
{
	// !!!
	
	for(int i = 1; i < _layerSizes.size(); i++)
	{
	
		std::ifstream myWFile("NNData/weights" + std::to_string(i) + ".txt");
		std::ifstream myBFile("NNData/biases" + std::to_string(i) + ".txt");
		
		if(myWFile.good() && myBFile.good())
		{
			//files exist, we need to load them in.
			
			
			
			//weights
			std::string line;
			int index = 0;
			
			while(getline(myWFile, line))
			{
				VectorXf lineVector;
				std::string delimiter = ",";
				size_t pos = 0;
				while((pos = line.find(delimiter)) != std::string::npos)
				{
					lineVector.conservativeResize(lineVector.size()+1);
					lineVector(lineVector.size()-1) = std::stof(line.substr(0,pos));
				}
				lineVector.conservativeResize(lineVector.size()+1);
				lineVector(lineVector.size()-1) = std::stof(line);
				
				_weights[i].conservativeResize(_weights[i].rows(), _weights[i].cols()+1);
				_weights[i].col(index) = lineVector;
				
				index++;
			}
			
			
			//biases
			std::string lineB;
			
			getline(myBFile, lineB);
			
			VectorXf bVector;
			
			std::string delimiter2 = ",";
			size_t pos = 0;
			
			while((pos = lineB.find(delimiter2)) != std::string::npos)
			{
				bVector.conservativeResize(bVector.size()+1);
				bVector(bVector.size()-1) = std::stof(lineB.substr(0,pos));
			bVector.conservativeResize(bVector.size()+1);
			}
			bVector(bVector.size()-1) = std::stof(lineB);
			
			_biases[i] = bVector;			
		}
		else
		{
			//files don't exist, generate random numbers
			
			_weights[i] = MatrixXf::Random(_layerSizes[i], _layerSizes[i-1]);
			_biases[i] = VectorXf::Random(_layerSizes[i]);
			
		}
		myWFile.close();
		myBFile.close();
	
	}
}


int NeuralNetwork::getOutput(VectorXf a)
{
	VectorXf newA = a;
	
	for(int i = 1; i < _layerSizes.size(); i++)
	{
		newA = sigmoid(_weights[i]*newA + _biases[i]);
	}
	
	for(int i = 0; i < newA.size(); i++)
	{
		if(newA(i) == newA.maxCoeff())
			return i;
	}
	
	return -1;
}


void NeuralNetwork::stochasticGradientDescent(std::vector<VectorXf> trainingSet, std::vector<VectorXf> trainingAnswer, float learningRate)
{
	/********************************************************************************************************************************
	
	My inner math geek would not let me implement this algorithm without justifying it. Probably could be more rigorous, but this
	will have to do for now.
	
	Our goal when training our neural network is to increase the accuracy of our neural net's predictions. In other words, we want to
	minimize the amount of errors in our neural net. In order to accomplish this we will define a cost function, which returns a 'high'
	value when it is failing, and a 'low' value when it is succeeding.
	
	More precisely, we will define a cost function for a fixed training example C = \frac{1}{2}\sum_{j=1}^{n}(y_j - a_j^l)^2.
	Now, we want to find a minimum for this cost function. We will accomplish this with gradient descent. The gradient of the cost
	function gives us the direction of 'steepest ascent', so taking the negative of the gradient of the cost function gives us the
	direction of 'steepest descent'. We then want to move in that direction and take the gradient again, until we find some local
	minimum. 
	
	Our cost function is really a function of two variables -- which we have control over -- our weights, and our biases. Hence, by
	definition, the gradient of the cost function will be given by the partial derivative of the cost function with respect to our
	weights, and the partial derivative of the cost function with respect to our biases. Let us now find these partial derivatives.
	
	Consider our cost function, C = \frac{1}{2}\sum_{j=1}^{n}(y_j - a_j^l)^2. Since in our implementation,
	a_j^l = sigmoid(\sum_{k=1}^n (w_{jk}^l a_k^{l-1}) + b_j^{l}), we can reexpress the cost function as
	C = \frac{1}{2}\sum_{j=1}^{n}(y_j - sigmoid(\sum_{k=1}^m (w_{jk}^l a_k^{l-1}) + b_j^{l}))^2.
	Now denote z_j^l = \sum_{k=1}^m w_{jk} a_k^{l-1} + b_j^l. Then a_j^l = sigmoid(z_j^l).
	
	Now we can use the chain rule to find our partial derivatives. dC/dw_{jk}^l = (dC/dz_j^l)(dz_j^l/dw_{jk}^l) = (dC/dz_j^l)(a_k^{l-1}),
	and dC/db_j^l = (dC/dz_j^l)(dz_j^l/db_j^l) = dC/dz_j^l.
	
	We are getting somewhere now. Now, since both of our partial derivatives above are dependent on dC/dz_j^l, this seems to be the next
	natural thing to calculate. Again, by the chain rule, dC/dz_j^l = \sum_{k=1}^{p}(dC/dz_k^(l+1))(dz_k^(l+1)/dz_j^l). But 
	(dz_k^(l+1)/dz_j^l) = w_{kj}^(l+1)sigma'(z_j^l). So  dC/dz_j^l = \sum_{j=1}^{p}w_{kj}^{l+1}(dC/dz_k^(l+1))sigma'(z_j^l). Putting this
	into a vector, we get (dC/dz^l) = (((w^(l+1))^T (dC/dz^(l+1))) (H.Product) sigma'(z^l)). 
	
	So, with this new equation, we can find the gradient for the cost function on all layers if we can find (dC/dz^l) for the last layer,
	l=L. This is fairly straightforward. (dC/dz_j^L) = (dC/da_j^L)(da_j^L/dz_j^L) = (dC/da_j^L)sigmoid'(z_j^L) Therefore, all of our
	equations are determined. 
	
	
	********************************************************************************************************************************/
	
	
	
	
	
	// activation vector where each element is an activation vector for each element of the training set.
	std::vector<std::vector<VectorXf>> a;
	
	// vector of partial derivative of the cost function wrt z for each layer for each element x of the training set
	std::vector<std::vector<VectorXf>> dCdz;
	
	// loop through all x in training set
	for(int i = 0; i < trainingSet.size(); i++)
	{
		
		
		// vector of `z` (weighted input) where each element is the `z` corresponding to a layer.
		std::vector<VectorXf> z_x;
		z_x.resize(_layerSizes.size());
		// vector dCdz for x in training set
		std::vector<VectorXf> dCdz_x;
		dCdz_x.resize(_layerSizes.size());
		//vector a for x in the training set.
		std::vector<VectorXf> a_x;
		a_x.resize(_layerSizes.size());
		
		// set input layer activation vector
		a_x[0] = trainingSet[i];
		
		// z_x and DCdz_x not defined for initial layer, so set to zero.
		z_x[0] = VectorXf::Zero(1);
		dCdz_x[0] = VectorXf::Zero(1);
		
		// loop through layers and set z_x and a_x.
		for(int j = 1; j < _layerSizes.size(); j++)
		{
			z_x[j] = _weights[j]*a_x[j-1] + _biases[j];

			a_x[j] = sigmoid(z_x[j]);
		}
		
		// set dCdz for the last layer
		dCdz_x[_layerSizes.size()-1] = (a_x[_layerSizes.size()-1] - trainingAnswer[i]).cwiseProduct(sigmoidPrime(z_x[_layerSizes.size()-1]));
		
		// set dCdz for the rest of the layers.
		for(int j = _layerSizes.size()-2; j > 0; j++)
		{
			dCdz_x[j] = (_weights[j+1].transpose()*dCdz_x[j+1]).cwiseProduct(sigmoidPrime(z_x[j]));
		}
		
		dCdz.push_back(dCdz_x);
		a.push_back(a_x);
	}
	
	// most of the mathematical heavy-lifting is over. All that is left to do is adjust the weights and biases based on the average 
	// gradient values from the training set.
	
	for(int i = 1; i < _layerSizes.size(); i++)
	{
		VectorXf floatAdjustmentWSum = dCdz[0][i]*(a[0][i-1].transpose());
		for(int j = 1; i < trainingSet.size(); i++)
		{
			floatAdjustmentWSum = floatAdjustmentWSum + (dCdz[j][i]*(a[j][i-1].transpose()));
		}
		
		_weights[i] = _weights[i] - (learningRate/trainingSet.size())*floatAdjustmentWSum;
		
		VectorXf floatAdjustmentBSum = dCdz[0][i];
		for(int j = 1; i < trainingSet.size(); i++)
		{
			floatAdjustmentBSum = floatAdjustmentBSum + dCdz[j][i];
		}
		
		_biases[i] = _biases[i] - (learningRate/trainingSet.size())*floatAdjustmentBSum;
	}
	
	// all done!
}


VectorXf NeuralNetwork::sigmoid(VectorXf z)
{
	// f(x) = \frac{1}{1 + e^{-x}} - creates a smooth function ranging from 0 to 1 over R
	VectorXf newZ = z;
	
	for(int i = 0; i < z.size(); i++)
	{
		newZ(i) = 1/(1 + std::exp(-z(i)));
	}
	
	return newZ;
}


VectorXf NeuralNetwork::sigmoidPrime(VectorXf z)
{
	// f'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} 
	VectorXf newZ = z;
	
	for(int i = 0; i < z.size(); i++)
	{
		newZ(i) = (std::exp(-z(i)))/(std::pow(2.0, (1 + std::exp(-z(i)))));
	}
	
	return newZ;
}


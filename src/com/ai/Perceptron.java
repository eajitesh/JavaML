package com.ai;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * This is a Java implementation of RosenBlatt Percepton which is also called as 
 * single-layer neural network. The implementation will comprise of the following:
 * 
 * 1. Calculate net input: Net input is a weighted sum of input
 * 2. Predict the class based on the net input
 * 3. Update the weights
 * 4. Loop through 1, 2, 3 for number of iterations (epochs)
 * 
 * Note that the unit step function acts as an activation function in this simplistic implementation of Percepton.
 * 
 * Lets say Z = w1x1 + w2x2 +... + wnxn
 * 
 * 		{ 1 if Z >= theta
 * Y =  {
 * 		{ -1 if Z < theta
 * 
 * @author ajitesh.kumar
 *
 */
public class Perceptron {
	
	private static final double DefaultLearningRate = 0.01;
	private static final int DefaultNoOfIterations = 50;
	
	private int noOfIteration = 0;
	private double learningRate = 0.0;
	private double[] weights;
	private List<Integer> errors;
	
	public Perceptron() {
		this.learningRate = DefaultLearningRate;
		this.noOfIteration = DefaultNoOfIterations;
		this.weights = null;
		this.errors = new ArrayList<Integer>();
	}
	
	public Perceptron(int noOfIteration) {
		this.noOfIteration = noOfIteration;
		this.learningRate = DefaultLearningRate;
		this.weights = null;
		this.errors = new ArrayList<Integer>();
	}
	
	public Perceptron(int noOfIteration, double learningRate) {
		this.noOfIteration = noOfIteration;
		this.learningRate = learningRate;
		this.weights = null;
		this.errors = new ArrayList<Integer>();
	}
	
	public double netInput(double[] input) {
		double netInput = 0;
		netInput += this.weights[0];
		for(int i = 0; i < input.length; i++) {
			netInput += this.weights[i+1]*input[i];			
		}
		return netInput;
	}
	
	private int activationFunction(double[] input) {
		double netInput = this.netInput(input);
		if(netInput >= 0) {
			return 1;
		}
		return -1;
	}
	
	public int predict(double[] input) {
		
		return activationFunction(input);
	}
	

	public void fit(double[][] x, double[] y) {
		//
		// Initialize the weights
		//
		Random rd = new Random();
		this.weights = new double[x[0].length + 1];
		for(int i = 0; i < this.weights.length; i++) {
			this.weights[i] = rd.nextDouble();
		}
		for(int i = 0; i < this.weights.length; i++) {
			System.out.println("Weights :" + this.weights[i]);
		}
		//
		// Fit the model
		//		
		for(int i = 0; i < this.noOfIteration; i++) {
			int errorInEachIteration = 0;
			for(int j=0; j < x.length; j++) {
				//
				// Calculate the output of activation function for each input
				//
				double activationFunctionOutput = activationFunction(x[j]);
				this.weights[0] += this.learningRate*(y[j] - activationFunctionOutput);
				for(int k = 0; k < x[j].length; k++) {
					//
					// Calculate the delta weight which needs to be updated 
					// for each feature
					//
					double deltaWeight = this.learningRate*(y[j] - activationFunctionOutput)*x[j][k];
					this.weights[k+1] += deltaWeight;
				}
				//
				// Calculate error for each training data
				//
				if(y[j] != this.predict(x[j])) {
					errorInEachIteration++;
				}
			}
			//
			// Update the error in each Epoch
			//
			this.errors.add(errorInEachIteration);
		}
		for(int l = 0; l < this.weights.length; l++) {
			System.out.println("Final Weights :" + this.weights[l]);
		}
	}
	
	public void printErrors() {
		Iterator<Integer> iter = this.errors.iterator();
		int i = 1;
		while(iter.hasNext()) {
			System.out.println(iter.next());
		}
	}
	
	private void printWeights() {
		for(int i = 0; i < this.weights.length; i++) {
			System.out.println("Weight " + i +": " + this.weights[i]);
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		double[][] X = {{5.1, 1.4},{4.9, 1.4},{4.7, 1.3},{4.6, 1.5},{5. , 1.4},{5.4, 1.7},{4.6, 1.4},{5. , 1.5},{4.4, 1.4},{4.9, 1.5},{5.4, 1.5},{4.8, 1.6},{4.8, 1.4},{4.3, 1.1},{5.8, 1.2},{5.7, 1.5},{5.4, 1.3},{5.1, 1.4},{5.7, 1.7},{5.1, 1.5},{5.4, 1.7},{5.1, 1.5},{4.6, 1. },{5.1, 1.7},{4.8, 1.9},{5. , 1.6},{5. , 1.6},{5.2, 1.5},{5.2, 1.4},{4.7, 1.6},{4.8, 1.6},{5.4, 1.5},{5.2, 1.5},{5.5, 1.4},{4.9, 1.5},{5. , 1.2},{5.5, 1.3},{4.9, 1.5},{4.4, 1.3},{5.1, 1.5},{5. , 1.3},{4.5, 1.3},{4.4, 1.3},{5. , 1.6},{5.1, 1.9},{4.8, 1.4},{5.1, 1.6},{4.6, 1.4},{5.3, 1.5},{5. , 1.4},{7. , 4.7},{6.4, 4.5},{6.9, 4.9},{5.5, 4. },{6.5, 4.6},{5.7, 4.5},{6.3, 4.7},{4.9, 3.3},{6.6, 4.6},{5.2, 3.9},{5. , 3.5},{5.9, 4.2},{6. , 4. },{6.1, 4.7},{5.6, 3.6},{6.7, 4.4},{5.6, 4.5},{5.8, 4.1},{6.2, 4.5},{5.6, 3.9},{5.9, 4.8},{6.1, 4. },{6.3, 4.9},{6.1, 4.7},{6.4, 4.3},{6.6, 4.4},{6.8, 4.8},{6.7, 5. },{6. , 4.5},{5.7, 3.5},{5.5, 3.8},{5.5, 3.7},{5.8, 3.9},{6. , 5.1},{5.4, 4.5},{6. , 4.5},{6.7, 4.7},{6.3, 4.4},{5.6, 4.1},{5.5, 4. },{5.5, 4.4},{6.1, 4.6},{5.8, 4. },{5. , 3.3},{5.6, 4.2},{5.7, 4.2},{5.7, 4.2},{6.2, 4.3},{5.1, 3. },{5.7, 4.1}};
		double[] Y = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		              -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,
		               1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
		               1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
		               1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1};
		
		Perceptron prcptrn = new Perceptron(10);
		prcptrn.fit(X, Y);
		prcptrn.printErrors();
		prcptrn.printWeights();
	}
}

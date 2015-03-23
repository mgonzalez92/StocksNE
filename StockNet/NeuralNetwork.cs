using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StockNet
{
    class NeuralNetwork
    {
        // Number of neurons
        private int inputNum;
        private int hiddenNum;
        private int outputNum;

        // Weights
        public double[][] hiddenWeights;
        public double[] hiddenBiases;
        public double[][] outputWeights;
        public double[] outputBiases;

        // Neuron values
        public double[] inputValues;
        public double[] hiddenValues;
        public double[] outputValues;

        public double fitness;

        private Random random;

        public NeuralNetwork(int inputNum, int hiddenNum, int outputNum, Random random)
        {
            this.inputNum = inputNum;
            this.hiddenNum = hiddenNum;
            this.outputNum = outputNum;
            this.random = random;

            // Initialize neuron value arrays
            inputValues = new double[inputNum];
            hiddenValues = new double[hiddenNum];
            outputValues = new double[outputNum];

            // Initialize weight arrays
            hiddenWeights = new double[hiddenNum][];
            hiddenBiases = new double[hiddenNum];
            outputWeights = new double[outputNum][];
            outputBiases = new double[outputNum];

            // Initialize random weights
            for (int i = 0; i < hiddenNum; i++)
            {
                hiddenWeights[i] = new double[inputNum];
                for (int j = 0; j < inputNum; j++)
                {
                    hiddenWeights[i][j] = random.NextDouble();
                }
                hiddenBiases[i] = random.NextDouble();
            }
            for (int i = 0; i < outputNum; i++)
            {
                outputWeights[i] = new double[hiddenNum];
                for (int j = 0; j < hiddenNum; j++)
                {
                    outputWeights[i][j] = random.NextDouble();
                }
                outputBiases[i] = random.NextDouble();
            }
        }

        public void FeedForward()
        {
            double sum;

            // Compute hidden layer
            for (int i = 0; i < hiddenNum; i++)
            {
                sum = 0;
                for (int j = 0; j < inputNum; j++)
                {
                    sum += inputValues[j] * hiddenWeights[i][j];
                }
                hiddenValues[i] = 1.0 / (1.0 + Math.Exp(-(sum - hiddenBiases[i])));
            }

            // Compute output layer
            for (int i = 0; i < outputNum; i++)
            {
                sum = 0;
                for (int j = 0; j < hiddenNum; j++)
                {
                    sum += hiddenValues[j] * outputWeights[i][j];
                }
                outputValues[i] = 1.0 / (1.0 + Math.Exp(-(sum - outputBiases[i])));
            }
        }

        public double[] FlattenWeights()
        {
            double[] weights = new double[hiddenNum * (inputNum + 1) + outputNum * (hiddenNum + 1)];
            int c = 0;

            for (int i = 0; i < hiddenNum; i++)
            {
                for (int j = 0; j < inputNum; j++)
                {
                    weights[c] = hiddenWeights[i][j];
                    c++;
                }
                weights[c] = hiddenBiases[i];
                c++;
            }
            for (int i = 0; i < outputNum; i++)
            {
                for (int j = 0; j < hiddenNum; j++)
                {
                    weights[c] = outputWeights[i][j];
                    c++;
                }
                weights[c] = outputBiases[i];
                c++;
            }

            return weights;
        }

        public void InflateGenes(double[] weights)
        {
            int c = 0;

            for (int i = 0; i < hiddenNum; i++)
            {
                for (int j = 0; j < inputNum; j++)
                {
                    hiddenWeights[i][j] = weights[c];
                    c++;
                }
                hiddenBiases[i] = weights[c];
                c++;
            }
            for (int i = 0; i < outputNum; i++)
            {
                for (int j = 0; j < hiddenNum; j++)
                {
                    outputWeights[i][j] = weights[c];
                    c++;
                }
                outputBiases[i] = weights[c];
                c++;
            }
        }

    }
}

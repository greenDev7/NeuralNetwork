using System;

namespace NeuralNetwork
{
    public class Neuron
    {
        /// <summary>
        /// Массив весовых коэффициентов (синаптические веса)
        /// </summary>
        private double[] Weights { get; }
        /// <summary>
        /// Пороговый элемент (стр. 43)
        /// </summary>
        private double Bias { get; }
        /// <summary>
        /// Функция активации
        /// </summary>
        private Func<double, double> ActivationFunction { get; }


        public Neuron(int Dimension, Func<double, double> ActivationFunction, double leftRangeLimit = -1.0, double rightRangeLimit = 1.0, double Bias = 0.0)
        {
            this.ActivationFunction = ActivationFunction;
            this.Bias = Bias;

            this.Weights = new double[Dimension];

            Random random = new Random();

            for (int i = 0; i < Dimension; i++)
                Weights[i] = random.NextDouble() * (rightRangeLimit - leftRangeLimit) + leftRangeLimit;
        }

        private double Adder(double[] inputSignals)
        {
            if (inputSignals.Length != this.Weights.Length)
                throw new ArgumentOutOfRangeException("inputSignals", "Количество входных сигналов не равно количеству весовых коэффициентов");

            double linearCombinerOutput = 0.0;

            for (int i = 0; i < inputSignals.Length; i++)
                linearCombinerOutput += inputSignals[i] * this.Weights[i];

            return linearCombinerOutput + this.Bias;
        }

        public double GetActivationPotential(double[] inputSignals)
        {
            double linearCombinerOutput = Adder(inputSignals);
            return this.ActivationFunction(linearCombinerOutput);
        }
    }
}

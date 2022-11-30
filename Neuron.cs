using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        /// <summary>
        /// Массив весовых коэффициентов (синаптические веса)
        /// </summary>
        private List<double> Weights { get; }
        /// <summary>
        /// Пороговый элемент (стр. 43)
        /// </summary>
        private double Bias { get; }
        /// <summary>
        /// Функция активации
        /// </summary>
        private Func<double, double> ActivationFunction { get; }


        public Neuron(Func<double, double> ActivationFunction, List<double> Weights = null, double Bias = 0.0)
        {
            this.Weights = Weights;
            this.ActivationFunction = ActivationFunction;
            this.Bias = Bias;
        }

        private double Adder(List<double> inputSignals)
        {
            if (inputSignals.Count != Weights.Count)
                throw new ArgumentOutOfRangeException("inputSignals", "Количество входных сигналов не равно количеству весовых коэффициентов");

            double linearCombinerOutput = 0.0;           

            for (int i = 0; i < inputSignals.Count; i++)
                linearCombinerOutput += inputSignals[i] * Weights[i];

            return linearCombinerOutput + Bias;
        }
        public double GetActivationPotential(List<double> inputSignals)
        {
            double linearCombinerOutput = Adder(inputSignals);
            return ActivationFunction(linearCombinerOutput);
        }
    }
}

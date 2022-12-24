using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public double Bias { get; }
        private Func<double, double> ActivationFunction { get; }
        public double? InducedLocalField { get; private set; }
        public double LocalGradient { get; private set; }

        public Neuron(Func<double, double> ActivationFunction, List<double> Weights = null, double Bias = 0.0)
        {
            this.ActivationFunction = ActivationFunction;            
            this.Weights = Weights;
            this.Bias = Bias;
            this.InducedLocalField = null;
        }        
        
        public void SetInducedLocalField(List<double> inputSignal)
        {
            if (inputSignal.Count != Weights.Count)
                throw new ArgumentOutOfRangeException("inputSignals", "Ошибка при вычислении индуцированного локального поля: количество входных сигналов не равно количеству весовых коэффициентов");

            InducedLocalField = 0.0;

            for (int i = 0; i < inputSignal.Count; i++)
                InducedLocalField += inputSignal[i] * Weights[i];

            InducedLocalField += Bias;
        }

        internal void SetLocalGradient(double localGradient) => LocalGradient = localGradient;

        internal void AdjustWeights(double learningRateParameter, List<double> inputSignals)
        {
            for (int i = 0; i < Weights.Count; i++)
                Weights[i] += learningRateParameter * LocalGradient * inputSignals[i];
        }

        public double GetActivationPotential()
        {
            if (InducedLocalField == null)
                throw new ArgumentNullException("InducedLocalField", "В функцию активации передан аргумент равный null");
            else
                return ActivationFunction((double)InducedLocalField);
        }        
    }
}
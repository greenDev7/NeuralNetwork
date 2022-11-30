using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public LayerType LayerType { get; set; }

        private List<double> _inputSignals;
        public List<double> InputSignals
        {
            get => _inputSignals;

            set
            {
                if (this.LayerType != LayerType.Input)
                    throw new Exception("Входные сигналы можно задать только для входного слоя!");

                _inputSignals = value;
            } 
        }

        public Layer(List<Neuron> Neurons, LayerType LayerType)
        {
            this.Neurons = Neurons;
            this.LayerType = LayerType;
        }

        internal List<double> ProduceSignals() => InputSignals;

        internal List<double> ProduceSignals(List<double> previousLayerSignals)
        {
            List<double> currentSignals = new List<double>();

            foreach (Neuron neuron in Neurons)
                currentSignals.Add(neuron.GetActivationPotential(previousLayerSignals));
       
            return currentSignals;
        }
    }
}
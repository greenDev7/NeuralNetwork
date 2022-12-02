using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public LayerType LayerType { get; set; }

        public Layer(List<Neuron> Neurons, LayerType LayerType)
        {
            this.Neurons = Neurons;
            this.LayerType = LayerType;
        }

        internal List<double> ProduceSignals(List<double> previousLayerSignals)
        {
            List<double> currentSignals = new List<double>();

            foreach (Neuron neuron in Neurons)
                currentSignals.Add(neuron.GetActivationPotential(previousLayerSignals));
       
            return currentSignals;
        }
    }
}
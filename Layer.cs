using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }

        public Layer(List<Neuron> Neurons)
        {
            this.Neurons = Neurons;            
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
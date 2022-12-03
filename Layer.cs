using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        public LayerType LayerType { get; }
        public List<Neuron> Neurons { get; }

        public Layer(LayerType LayerType, List<Neuron> Neurons)
        {
            this.LayerType = LayerType;
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
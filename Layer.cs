using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }

        public List<double> InputSignals { get; set; }

        public Layer(List<Neuron> Neurons)
        {
            this.Neurons = Neurons;            
        }

        internal List<double> ProduceSignals()
        {
            List<double> currentSignals = new List<double>();

            foreach (Neuron neuron in Neurons)
                currentSignals.Add(neuron.GetActivationPotential());
       
            return currentSignals;
        }

        internal void CalculateAndSetLocalGradients(List<double> errorSignal)
        {
            for (int i = 0; i < Neurons.Count; i++)
                Neurons[i].SetLocalGradient(errorSignal[i] * ActivationFunctions.LogisticFunctionsDerivative((double)Neurons[i].InducedLocalField));
        }
        
        internal void CalculateAndSetLocalGradients(Layer previousLayer)
        {

            for (int i = 0; i < Neurons.Count; i++)
            {
                List<double> associatedWeights = GetAssociatedWeights(previousLayer, i);

                double innerSum = GetInnerSum(associatedWeights, previousLayer);

                Neurons[i].SetLocalGradient(innerSum * ActivationFunctions.LogisticFunctionsDerivative((double)Neurons[i].InducedLocalField));
            }
        }

        private double GetInnerSum(List<double> associatedWeights, Layer previousLayer)
        {
            double innerSum = 0.0;

            for (int i = 0; i < associatedWeights.Count; i++)
                innerSum += associatedWeights[i] * previousLayer.Neurons[i].LocalGradient;

            return innerSum;
        }

        private List<double> GetAssociatedWeights(Layer previousLayer, int neuronPositionInCurrentLayer)
        {
            List<double> associatedWeights = new List<double>();

            foreach (Neuron neuron in previousLayer.Neurons)
                associatedWeights.Add(neuron.Weights[neuronPositionInCurrentLayer]);

            return associatedWeights;
        }

        internal void AdjustWeights(double learningRateParameter)
        {
            for (int i = 0; i < Neurons.Count; i++)
                Neurons[i].AdjustWeights(learningRateParameter, InputSignals);
        }
    }
}
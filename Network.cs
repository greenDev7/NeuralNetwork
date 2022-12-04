using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        public List<Layer> HiddenLayers { get; }
        public Layer OutputLayer { get; }

        public Network(List<Layer> HiddenLayers, Layer OutputLayer)
        {
            this.HiddenLayers = HiddenLayers;
            this.OutputLayer = OutputLayer;
        }

        public List<double> PropagateForward(List<double> functionSignal)
        {
            // Передаем сигнал по скрытым слоям
            foreach (Layer hiddenLayer in HiddenLayers)
                functionSignal = hiddenLayer.ProduceSignals(functionSignal);

            // Возвращаем сигнал от выходного слоя
            return OutputLayer.ProduceSignals(functionSignal);
        }
        public void WriteHiddenWeightsToCSVFile(string fileName)
        {
            TextWriter textWriter = new StreamWriter(fileName);

            textWriter.WriteLine(string.Join(";", "hiddenLayersCount", HiddenLayers.Count));
            textWriter.WriteLine(string.Format("{0};{1}", "hiddenLayersDimensions", string.Join(";", HiddenLayers.Select(x => x.Neurons.Count))));

            foreach (Layer hiddenLayer in HiddenLayers)
                foreach (Neuron neuron in hiddenLayer.Neurons)
                    textWriter.WriteLine(string.Join(";", neuron.Weights));

            textWriter.Close();
        }
        public void WriteOutputWeightsToCSVFile(string fileName)
        {
            TextWriter textWriter = new StreamWriter(fileName);

            foreach (Neuron neuron in OutputLayer.Neurons)
                textWriter.WriteLine(string.Join(";", neuron.Weights));

            textWriter.Close();
        }
    }
}
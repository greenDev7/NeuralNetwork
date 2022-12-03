using System.Collections.Generic;

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

        public List<double> PropagateForward(List<double> signals)
        {
            // Передаем сигналы по скрытым слоям
            foreach (Layer hiddenLayer in HiddenLayers)
                signals = hiddenLayer.ProduceSignals(signals);

            // Возвращаем сигналы от выходного слоя
            return OutputLayer.ProduceSignals(signals);
        }
    }
}
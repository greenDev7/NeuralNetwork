using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Network
    {
        public Layer InputLayer { get; }        
        public List<Layer> HiddenLayers { get; }
        public Layer OutputLayer { get; }

        public Network(Layer InputLayer, List<Layer> HiddenLayers, Layer OutputLayer)
        {
            this.InputLayer = InputLayer;            
            this.HiddenLayers = HiddenLayers;
            this.OutputLayer = OutputLayer;
        }

        public List<double> PropagateForward()
        {
            // Получаем сигналы от входного слоя
            List<double> signals = InputLayer.ProduceSignals();

            // Передаем сигналы по скрытым слоям
            foreach (Layer hiddenLayer in HiddenLayers)
                signals = hiddenLayer.ProduceSignals(signals);

            // Возвращаем сигналы от выходного слоя
            return OutputLayer.ProduceSignals(signals);
        }
    }
}
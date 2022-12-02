using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
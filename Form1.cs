using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();            
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            string docPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);

            #region Инициализируем скрытые слои

            // Читаем весовые коэффициенты из файла
            List<List<List<double>>> hiddenLayersWeights =
                WeightsReader.ReadHiddenLayersWeightsFromCSVFile(Path.Combine(docPath, "hiddenLayersXORProblem.csv"));

            // Формируем скрытые слои
            /// TODO: Можно добавить массив с функциями активаций, чтобы каждый скрытый слой имел свою функцию активации 
            List<Layer> hiddenLayers = new List<Layer>();
            foreach (var layer in hiddenLayersWeights)
            {
                List<Neuron> neurons = new List<Neuron>();

                foreach (var weights in layer)
                    neurons.Add(new Neuron(ActivationFunctions.ThresholdFunction, Weights: weights.Skip(1).ToList(), Bias: weights[0]));

                hiddenLayers.Add(new Layer(neurons));
            }

            #endregion

            #region Инициализируем выходной слой

            // Читаем весовые коэффициенты из файла
            List<List<double>> outputLayerWeights = WeightsReader.ReadOutputLayerWeightsFromCSVFile(Path.Combine(docPath, "outputWeightsXORProblem.csv"));

            // Формируем выходной слой
            List<Neuron> outputNeurons = new List<Neuron>();
            for (int i = 0; i < outputLayerWeights.Count; i++)
                outputNeurons.Add(new Neuron(ActivationFunctions.ThresholdFunction, Weights: outputLayerWeights[i].Skip(1).ToList(), Bias: outputLayerWeights[i][0]));

            Layer outputLayer = new Layer(outputNeurons);
            #endregion

            // Инициализируем нейросеть с помощью слоев (скрытых и выходного)
            Network network = new Network(hiddenLayers, outputLayer);

            // Формируем входной сигнал
            List<double> functionSignal1 = new List<double> { 0.0, 1.0 };
            List<double> functionSignal2 = new List<double> { 1.0, 0.0 };

            List<double> functionSignal3 = new List<double> { 1.0, 1.0 };
            List<double> functionSignal4 = new List<double> { 0.0, 0.0 };

            // Прогоняем входной сигнал через нейросеть и получаем сигнал на выходе
            List<double> outputSignal1 = network.PropagateForward(functionSignal1);
            List<double> outputSignal2 = network.PropagateForward(functionSignal2);
            List<double> outputSignal3 = network.PropagateForward(functionSignal3);
            List<double> outputSignal4 = network.PropagateForward(functionSignal4);

            // Записываем весовые коэффициенты в файлы
            //network.WriteHiddenWeightsToCSVFile(Path.Combine(docPath, "hiddenLayers.csv"));
            //network.WriteOutputWeightsToCSVFile(Path.Combine(docPath, "outputWeights.csv"));
        }
    }
}
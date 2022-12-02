using System;
using System.Collections.Generic;
using System.IO;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();            
        }

        private void Form1_Load(object sender, System.EventArgs e)
        {
            string docPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);

            // Инициализируем скрытые слои
            int hiddenLayersCount = 2;
            int[] hiddenLayersDimensions = new int[2] { 3, 2 };
            List<List<List<double>>> hiddenLayersWeights =
                WeightsReaderWriter.ReadHiddenLayersWeightsFromCSVFile(hiddenLayersCount, hiddenLayersDimensions, Path.Combine(docPath, "weightsOfHiddenLayers.csv"));

            // Инициализируем выходной слой
            int outputLayerDimension = 2; // Количество нейронов на выходном слое
            List<List<double>> outputLayerWeights = WeightsReaderWriter.ReadOutputLayerWeightsFromCSVFile(Path.Combine(docPath, "weightsOfOutputLayer.csv"));


            int inputLayerDimension = 3; // Количество "нейронов" во входном слое 
            List<double> inputSignals = new List<double>();
            inputSignals = new List<double> { 0.0, 0.0, 1.0 };


            List<Neuron> inputNeurons = new List<Neuron>();
            List<Neuron> outputNeurons = new List<Neuron>();
            List<Neuron> firstHiddenLayerNeurons = new List<Neuron>();

            for (int i = 0; i < inputLayerDimension; i++)
                inputNeurons.Add(new Neuron(ActivationFunctions.Func1));

            for (int i = 0; i < outputLayerDimension; i++)
                outputNeurons.Add(new Neuron(ActivationFunctions.Func1, Weights: outputLayerWeights[i]));

            //for (int i = 0; i < hiddenLayersDimensions[0]; i++)
            //    firstHiddenLayerNeurons.Add(new Neuron(ActivationFunctions.Func1, Weights: firstHiddenLayerWeights[0]));


            Layer inputLayer = new Layer(inputNeurons, LayerType.Input);
            Layer outputLayer = new Layer(outputNeurons, LayerType.Output);
            Layer firstHiddenLayer = new Layer(firstHiddenLayerNeurons, LayerType.Hidden);

            Network network = new Network(inputLayer, new List<Layer> { firstHiddenLayer }, outputLayer);
        }
    }
}

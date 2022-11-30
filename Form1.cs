using System.Collections.Generic;
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
            int inputLayerDimension = 3; // Размерность (количество "нейронов") входного слоя 
            int outputLayerDimension = 2; // Размерность выходного слоя

            int hiddenLayersCount = 1; // Количество скрытых слоев
            int[] hiddenLayerDimensions = new int[hiddenLayersCount]; // Массив размерностей скрытых слоев (количество нейронов на каждом скрытом слое)
            hiddenLayerDimensions[0] = 2;                         // Размерность первого скрытого слоя
            // hiddenLayerDimensions[1] = 8;                      // Размерность второго скрытого слоя
            // hiddenLayerDimensions[2] = 5;                      // Размерность третьего скрытого слоя
            // ...
            // и т.д.
            // ...
            // hiddenLayerDimensions[hiddenLayersCount - 1] = 10;  // Размерность последнего скрытого слоя

            List<double> inputSignals = new List<double>();
            List<List<double>> outputLayerWeights = new List<List<double>>();
            List<List<List<double>>> hiddenLayerWeights = new List<List<List<double>>>();

            inputSignals = new List<double> { 0.0, 0.0, 1.0 };
            outputLayerWeights = new List<List<double>>()
            {
                new List<double>() {-0.29, 0.73},
                new List<double>() { 0.89, -0.53 },
            };

            List<List<double>> firstHiddenLayerWeights = new List<List<double>>()
            {
                new List<double>() {0.17, -0.43, 0.68},
                new List<double>() {0.27, 0.93, -0.13}
            };           


            List<Neuron> inputNeurons = new List<Neuron>();
            List<Neuron> outputNeurons = new List<Neuron>();
            List<Neuron> firstHiddenLayerNeurons = new List<Neuron>();

            for (int i = 0; i < inputLayerDimension; i++)
                inputNeurons.Add(new Neuron(ActivationFunctions.Func1));

            for (int i = 0; i < outputLayerDimension; i++)
                outputNeurons.Add(new Neuron(ActivationFunctions.Func1, Weights: outputLayerWeights[i]));

            for (int i = 0; i < hiddenLayerDimensions[0]; i++)
                firstHiddenLayerNeurons.Add(new Neuron(ActivationFunctions.Func1, Weights: firstHiddenLayerWeights[0]));


            Layer inputLayer = new Layer(inputNeurons, LayerType.Input);
            Layer outputLayer = new Layer(outputNeurons, LayerType.Output);
            Layer firstHiddenLayer = new Layer(firstHiddenLayerNeurons, LayerType.Hidden);


            inputLayer.InputSignals = inputSignals;

            Network network = new Network(inputLayer, new List<Layer> { firstHiddenLayer }, outputLayer);
        }
    }
}

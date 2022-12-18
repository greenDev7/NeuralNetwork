using System;
using System.Collections.Generic;
using System.Drawing;
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
            string trainDir = Path.Combine(docPath, "TrainingSet", "2");

            //string[] files = Directory.GetFiles(zeroDir);


            //string two = files.First(x => x.Contains("two"));
            //System.IO.File.Move(two, Path.Combine(zeroDir, "new3.png"));


            //string[] catalogsWithDigits = Directory.GetDirectories(trainDir);


            //List<string> digitImagesList = new List<string>();

            //foreach (string catalog in catalogsWithDigits)
            //    digitImagesList.AddRange(Directory.GetFiles(catalog).ToList());


            //Random rnd = new Random();
            //string[] MyRandomArray = digitImagesList.OrderBy(x => rnd.Next()).ToArray();


            List<List<double>> imageMatrix = ImageHelper.ConvertImageToPixelMatrix(Path.Combine(trainDir, "52.png"));

            //List<double> imageSignal = ImageHelper.ConvertImageToFunctionSignal(Path.Combine(docPath, "5_1.png"));

            ImageHelper.WritePixelMatrixToCSVFile(imageMatrix, Path.Combine(docPath, "52.csv"));            

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

            // Инициализируем нейросеть с помощью заданных параметров

            //int hiddenLayersCount = 10;
            //int[] hiddenLayersDimensions = new int[hiddenLayersCount];
            //Func<double, double>[] hiddenActivationFunctions = new Func<double, double>[hiddenLayersCount];

            //for (int i = 0; i < hiddenLayersCount; i++)
            //{
            //    hiddenLayersDimensions[i] = 900;
            //    hiddenActivationFunctions[i] = ActivationFunctions.LogisticFunction;
            //};

            //Network network = new Network(900, 10, ActivationFunctions.LogisticFunction, hiddenLayersDimensions, hiddenActivationFunctions);

            List<double> inputSignal1 = new List<double>() { 0.0, 0.0 };
            List<double> inputSignal2 = new List<double>() { 0.0, 1.0 };
            List<double> inputSignal3 = new List<double>() { 1.0, 0.0 };
            List<double> inputSignal4 = new List<double>() { 1.0, 1.0 };

            List<double> outputSignal1 = network.PropagateForward(inputSignal1);
            List<double> outputSignal2 = network.PropagateForward(inputSignal2);
            List<double> outputSignal3 = network.PropagateForward(inputSignal3);
            List<double> outputSignal4 = network.PropagateForward(inputSignal4);

            // Записываем весовые коэффициенты в файлы
            //network.WriteHiddenWeightsToCSVFile(Path.Combine(docPath, "hiddenLayers.csv"));
            //network.WriteOutputWeightsToCSVFile(Path.Combine(docPath, "outputWeights.csv"));
        }      
    }
}
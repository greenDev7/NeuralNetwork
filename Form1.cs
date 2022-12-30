using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using MNIST.IO;

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

            string trainingDirectory = "";
            string testDirectory = "";

            string trainingImagesPath = Path.Combine(docPath, "train-images-idx3-ubyte");
            string trainingLabelsPath = Path.Combine(docPath, "train-labels-idx1-ubyte");

            string testImagesPath = Path.Combine(docPath, "t10k-images-idx3-ubyte");
            string testLabelsPath = Path.Combine(docPath, "t10k-labels-idx1-ubyte");

            ImageHelper.CreateImagesFromMnistFile(Path.Combine(docPath, "ImagesTest"), testImagesPath, testLabelsPath); 

            #region Инициализируем скрытые слои

            //// Читаем весовые коэффициенты из файла
            //List<List<List<double>>> hiddenLayersWeights =
            //    WeightsReader.ReadHiddenLayersWeightsFromCSVFile(Path.Combine(docPath, "hiddenLayersTest.csv"));

            //// Формируем скрытые слои
            ///// TODO: Можно добавить массив с функциями активаций, чтобы каждый скрытый слой имел свою функцию активации 
            //List<Layer> hiddenLayers = new List<Layer>();
            //foreach (var layer in hiddenLayersWeights)
            //{
            //    List<Neuron> neurons = new List<Neuron>();

            //    foreach (var weights in layer)
            //        neurons.Add(new Neuron(ActivationFunctions.LogisticFunction, Weights: weights.Skip(1).ToList(), Bias: weights[0]));

            //    hiddenLayers.Add(new Layer(neurons));
            //}

            #endregion

            #region Инициализируем выходной слой

            //// Читаем весовые коэффициенты из файла
            //List<List<double>> outputLayerWeights = WeightsReader.ReadOutputLayerWeightsFromCSVFile(Path.Combine(docPath, "outputWeightsTest.csv"));

            //// Формируем выходной слой
            //List<Neuron> outputNeurons = new List<Neuron>();
            //for (int i = 0; i < outputLayerWeights.Count; i++)
            //    outputNeurons.Add(new Neuron(ActivationFunctions.LogisticFunction, Weights: outputLayerWeights[i].Skip(1).ToList(), Bias: outputLayerWeights[i][0]));

            //Layer outputLayer = new Layer(outputNeurons);
            #endregion

            // Инициализируем нейросеть с помощью слоев (скрытых и выходного)
            // Network network = new Network(hiddenLayers, outputLayer);

            // Инициализируем нейросеть с помощью заданных параметров

            int hiddenLayersCount = 1;
            int[] hiddenLayersDimensions = new int[hiddenLayersCount];
            Func<double, double>[] hiddenActivationFunctions = new Func<double, double>[hiddenLayersCount];

            hiddenLayersDimensions[0] = 300;
            //hiddenLayersDimensions[1] = 50;

            hiddenActivationFunctions[0] = ActivationFunctions.LogisticFunction;
            //hiddenActivationFunctions[1] = ActivationFunctions.LogisticFunction;

            Network network = new Network(900, 10, ActivationFunctions.LogisticFunction, hiddenLayersDimensions, hiddenActivationFunctions);
            //Network network = new Network(900, 10, ActivationFunctions.LogisticFunction);

            double totalError;
            List<double> errorList = network.Train(trainingDirectory, out totalError, 0.1, 2);
            //List<double> errorList = network.TestTrain(out totalError, 0.5);



            WriteErrorListToCSVFile(errorList, Path.Combine(docPath, "errorList.csv"));


            List<(int, string)> randomTestSet = ImageHelper.GetRandomImagesPaths(testDirectory);

            foreach ((int, string) image in randomTestSet)
            {
                List<double> functionSignal = ImageHelper.ConvertImageToFunctionSignal(image.Item2);

                List<double> outputSignal = network.MakePropagateForward(functionSignal);
            }


            //List<double> inputSignal1 = new List<double>() { 0.0, 0.0 };
            //List<double> inputSignal2 = new List<double>() { 0.0, 1.0 };
            //List<double> inputSignal3 = new List<double>() { 1.0, 0.0 };
            //List<double> inputSignal4 = new List<double>() { 1.0, 1.0 };

            //List<double> outputSignal1 = network.MakePropagateForward(inputSignal1);
            //List<double> outputSignal2 = network.MakePropagateForward(inputSignal2);
            //List<double> outputSignal3 = network.MakePropagateForward(inputSignal3);
            //List<double> outputSignal4 = network.MakePropagateForward(inputSignal4);

            // Записываем весовые коэффициенты в файлы
            //network.WriteHiddenWeightsToCSVFile(Path.Combine(docPath, "hiddenLayer_res2.csv"));
            //network.WriteOutputWeightsToCSVFile(Path.Combine(docPath, "outputWeights_res2.csv"));
        }

        private void WriteErrorListToCSVFile(List<double> errorList, string fileName)
        {
            TextWriter textWriter = new StreamWriter(fileName);

            for (int i = 0; i < errorList.Count; i++)
                textWriter.WriteLine("{0};{1}", i, errorList[i]);          

            textWriter.Close();
        }
    }
}
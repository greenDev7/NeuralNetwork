using MNIST.IO;
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
            string myDocumentFolder = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);

            string trainingImagesPath = Path.Combine(myDocumentFolder, "train-images-idx3-ubyte");
            string trainingLabelsPath = Path.Combine(myDocumentFolder, "train-labels-idx1-ubyte");

            string testImagesPath = Path.Combine(myDocumentFolder, "t10k-images-idx3-ubyte");
            string testLabelsPath = Path.Combine(myDocumentFolder, "t10k-labels-idx1-ubyte");

            List<Layer> hiddenLayers = InitializeHiddenLayersWeightsFromCSVFile(Path.Combine(myDocumentFolder, "adjustedHiddenLayerWeights_acc9153.csv"));
            Layer outputLayer = InitializeOutputLayerWeightsFromCSVFile(Path.Combine(myDocumentFolder, "adjustedOutputLayerWeights_acc9153.csv"));

            Network network = new Network(hiddenLayers, outputLayer);

            #region Блок для инициализация нейросети рандомными значениями и ее обучение

            ////Инициализируем нейросеть с помощью заданных параметров

            //int hiddenLayersCount = 1;
            //int[] hiddenLayersDimensions = new int[hiddenLayersCount];
            //Func<double, double>[] hiddenActivationFunctions = new Func<double, double>[hiddenLayersCount];

            //hiddenLayersDimensions[0] = 40;
            //hiddenActivationFunctions[0] = ActivationFunctions.SigmoidFunction;

            //Network network = new Network(784, 10, ActivationFunctions.SigmoidFunction, hiddenLayersDimensions, hiddenActivationFunctions);
            //List<double> errorList = network.Train(trainingImagesPath, trainingLabelsPath, 0.2, 1);

            #endregion

            IEnumerable<TestCase> testCases = FileReaderMNIST.LoadImagesAndLables(testLabelsPath, testImagesPath);

            int incorrectPredictionsCount = 0;
            foreach (TestCase test in testCases)
            {
                List<double> functionSignal = ImageHelper.ConvertImageToFunctionSignal(test.Image);

                List<double> outputSignal = network.MakePropagateForward(functionSignal);

                int predictedDigit = outputSignal.IndexOf(outputSignal.Max());

                if (test.Label != predictedDigit)
                {
                    incorrectPredictionsCount++;
                    Bitmap bitmap = ImageHelper.CreateBitmapFromMnistImage(test.Image);
                    bitmap.Save(Path.Combine(myDocumentFolder, "IncorrectPredictions", $"{incorrectPredictionsCount}_{test.Label}_{predictedDigit}.png"));
                }
            }

            double accuracy = 100.0 - (incorrectPredictionsCount / 100.0);

            // Записываем оптимизированные весовые коэффициенты в файлы
            //network.WriteHiddenWeightsToCSVFile(Path.Combine(myDocumentFolder, "adjustedHiddenLayerWeights_accXX.csv"));
            //network.WriteOutputWeightsToCSVFile(Path.Combine(myDocumentFolder, "adjustedOutputLayerWeights_accXX.csv"));
        }

        /// <summary>
        /// Формирует выходной слой и инициализирует его весовые коэффициенты из CSV-файла
        /// </summary>
        /// <param name="fileName">путь к файлу с весовыми коэффициентами для выходного слоя</param>
        /// <returns>Выходной слой</returns>
        private Layer InitializeOutputLayerWeightsFromCSVFile(string fileName)
        {
            List<List<double>> outputLayerWeights = WeightsReader.ReadOutputLayerWeightsFromCSVFile(fileName);

            // Формируем выходной слой
            List<Neuron> outputNeurons = new List<Neuron>();
            for (int i = 0; i < outputLayerWeights.Count; i++)
                outputNeurons.Add(new Neuron(ActivationFunctions.SigmoidFunction, Weights: outputLayerWeights[i].Skip(1).ToList(), Bias: outputLayerWeights[i][0]));

            return new Layer(outputNeurons);
        }

        /// <summary>
        /// Формирует скрытые слои и инициализирует их весовые коэффициенты из CSV-файла
        /// </summary>
        /// <param name="fileName">путь к файлу с весовыми коэффициентами для скрытых слоев</param>
        /// <returns>Список скрытых слоев</returns>
        private List<Layer> InitializeHiddenLayersWeightsFromCSVFile(string fileName)
        {
            // Читаем весовые коэффициенты из файла
            List<List<List<double>>> hiddenLayersWeights = WeightsReader.ReadHiddenLayersWeightsFromCSVFile(fileName);

            // Формируем скрытые слои
            List<Layer> hiddenLayers = new List<Layer>();

            foreach (var layer in hiddenLayersWeights)
            {
                List<Neuron> neurons = new List<Neuron>();

                foreach (var weights in layer)
                    neurons.Add(new Neuron(ActivationFunctions.SigmoidFunction, Weights: weights.Skip(1).ToList(), Bias: weights[0]));

                hiddenLayers.Add(new Layer(neurons));
            }

            return hiddenLayers;
        }        
    }
}
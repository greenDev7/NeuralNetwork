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

        private void learnButton_Click(object sender, EventArgs e)
        {
            string myDocumentFolder = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);

            string trainingImagesPath = Path.Combine(myDocumentFolder, "train-images-idx3-ubyte");
            string trainingLabelsPath = Path.Combine(myDocumentFolder, "train-labels-idx1-ubyte");

            string testImagesPath = Path.Combine(myDocumentFolder, "t10k-images-idx3-ubyte");
            string testLabelsPath = Path.Combine(myDocumentFolder, "t10k-labels-idx1-ubyte");

            #region Блок для инициализации нейросети с помощью весовых коэффициентов из файлов csv

            // List<Layer> hiddenLayers = InitializeHiddenLayersWeightsFromCSVFile(Path.Combine(myDocumentFolder, "adjustedHiddenLayerWeights_acc9153.csv"));
            // Layer outputLayer = InitializeOutputLayerWeightsFromCSVFile(Path.Combine(myDocumentFolder, "adjustedOutputLayerWeights_acc9153.csv"));

            // Network network = new Network(hiddenLayers, outputLayer);

            #endregion


            #region Блок для инициализация нейросети рандомными значениями и ее обучение

            //Инициализируем нейросеть с помощью заданных параметров

            int hiddenLayersCount = 1;  // Задаем количество скрытых слоев
            int[] hiddenLayersDimensions = new int[hiddenLayersCount]; // Массив для хранения количества нейронов на каждом скрытом слое
            Func<double, double>[] hiddenActivationFunctions = new Func<double, double>[hiddenLayersCount]; // Массив для хранения функций активации на каждом скрытом слое

            hiddenLayersDimensions[0] = 35; // У нас один скрытый слой на котором 35 нейронов
            hiddenActivationFunctions[0] = ActivationFunctions.SigmoidFunction; // И для всех нейронов этого скрытого слоя используется сигмоидальная функция активации

            // 784 входа - это размер массива полученного из изображения(28 * 28 пикселей), 10 выходных нейронов
            Network network = new Network(784, 10, ActivationFunctions.SigmoidFunction, hiddenLayersDimensions, hiddenActivationFunctions);
            List<double> errorList = network.Train(trainingImagesPath, trainingLabelsPath, 0.2, 1); // Запускаем обучение

            #endregion


            #region Тестируем нейросеть на тестовой выборке в 10 000 изображений

            // Получаем тестовые изображения
            IEnumerable<TestCase> testCases = FileReaderMNIST.LoadImagesAndLables(testLabelsPath, testImagesPath);

            int incorrectPredictionsCount = 0; // счетчик неверно предсказанных результатов
            foreach (TestCase test in testCases)
            {
                List<double> functionSignal = ImageHelper.ConvertImageToFunctionSignal(test.Image); // Преобразуем изображение в вектор размерности 784 состоящий из нулей и единичек

                List<double> outputSignal = network.MakePropagateForward(functionSignal); // Получаем сигнал от нейросети

                int predictedDigit = outputSignal.IndexOf(outputSignal.Max()); // Предсказанную цифру находим как индекс максимального элемента массива

                // Если нейросеть выдала некорректный ответ
                if (test.Label != predictedDigit)
                {
                    incorrectPredictionsCount++;
                    //Bitmap bitmap = ImageHelper.CreateBitmapFromMnistImage(test.Image); // Получим это изображение
                    // И сохраним в папку IncorrectPredictions
                    //bitmap.Save(Path.Combine(myDocumentFolder, "IncorrectPredictions", $"{incorrectPredictionsCount}_{test.Label}_{predictedDigit}.png"));
                }
            }

            double accuracy = 100.0 - (incorrectPredictionsCount / 100.0); // Вычисляем точность (%)
            #endregion


            // Записываем скорректированные весовые коэффициенты в файлы
            network.WriteHiddenWeightsToCSVFile(Path.Combine(myDocumentFolder, "adjustedHiddenLayerWeights_accXX.csv"));
            network.WriteOutputWeightsToCSVFile(Path.Combine(myDocumentFolder, "adjustedOutputLayerWeights_accXX.csv"));
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
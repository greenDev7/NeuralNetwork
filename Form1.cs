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

            //List<List<double>> imageMatrix = ImageHelper.ConvertImageToPixelMatrix(Path.Combine(docPath, "3_test.png"));

            //ImageHelper.WritePixelMatrixToCSVFile(imageMatrix, Path.Combine(docPath, "3_test.csv"));            

            //#region Инициализируем скрытые слои

            //// Читаем весовые коэффициенты из файла
            //List<List<List<double>>> hiddenLayersWeights =
            //    WeightsReader.ReadHiddenLayersWeightsFromCSVFile(Path.Combine(docPath, "hiddenLayersXORProblem.csv"));

            //// Формируем скрытые слои
            ///// TODO: Можно добавить массив с функциями активаций, чтобы каждый скрытый слой имел свою функцию активации 
            //List<Layer> hiddenLayers = new List<Layer>();
            //foreach (var layer in hiddenLayersWeights)
            //{
            //    List<Neuron> neurons = new List<Neuron>();

            //    foreach (var weights in layer)
            //        neurons.Add(new Neuron(ActivationFunctions.ThresholdFunction, Weights: weights.Skip(1).ToList(), Bias: weights[0]));

            //    hiddenLayers.Add(new Layer(neurons));
            //}

            //#endregion

            //#region Инициализируем выходной слой

            //// Читаем весовые коэффициенты из файла
            //List<List<double>> outputLayerWeights = WeightsReader.ReadOutputLayerWeightsFromCSVFile(Path.Combine(docPath, "outputWeightsXORProblem.csv"));

            //// Формируем выходной слой
            //List<Neuron> outputNeurons = new List<Neuron>();
            //for (int i = 0; i < outputLayerWeights.Count; i++)
            //    outputNeurons.Add(new Neuron(ActivationFunctions.ThresholdFunction, Weights: outputLayerWeights[i].Skip(1).ToList(), Bias: outputLayerWeights[i][0]));

            //Layer outputLayer = new Layer(outputNeurons);
            //#endregion

            //// Инициализируем нейросеть с помощью слоев (скрытых и выходного)
            //Network network = new Network(hiddenLayers, outputLayer);

            // Инициализируем нейросеть с помощью заданных параметров
            Func<double, double>[] hiddenActivationFunctions = new Func<double, double>[]
            {
                ActivationFunctions.LogisticFunction,
                ActivationFunctions.ThresholdFunction,
                ActivationFunctions.LogisticFunction
            };

            Network network = new Network(10, 10, ActivationFunctions.ThresholdFunction, new int[] { 3, 5, 2 }, hiddenActivationFunctions);


            // Записываем весовые коэффициенты в файлы
            network.WriteHiddenWeightsToCSVFile(Path.Combine(docPath, "hiddenLayers.csv"));
            network.WriteOutputWeightsToCSVFile(Path.Combine(docPath, "outputWeights.csv"));
        }      
    }
}
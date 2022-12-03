﻿using System;
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

            #region Инициализируем скрытые слои

            // Читаем весовые коэффициенты из файла
            int hiddenLayersCount = 2;
            int[] hiddenLayersDimensions = new int[2] { 3, 2 };
            List<List<List<double>>> hiddenLayersWeights =
                WeightsReader.ReadHiddenLayersWeightsFromCSVFile(hiddenLayersCount, hiddenLayersDimensions, Path.Combine(docPath, "weightsOfHiddenLayers.csv"));

            // Формируем скрытые слои
            /// TODO: Можно добавить массив с функциями активаций, чтобы каждый скрытый слой имел свою функцию активации 
            List<Layer> hiddenLayers = new List<Layer>();
            foreach (var layer in hiddenLayersWeights)
            {
                List<Neuron> neurons = new List<Neuron>();

                foreach (var weights in layer)
                    neurons.Add(new Neuron(ActivationFunctions.Func1, weights));

                hiddenLayers.Add(new Layer(LayerType.Hidden, neurons));
            }

            #endregion

            #region Инициализируем выходной слой

            // Задаем количество нейронов на выходном слое
            int outputLayerDimension = 2;

            // Читаем весовые коэффициенты из файла
            List<List<double>> outputLayerWeights = WeightsReader.ReadOutputLayerWeightsFromCSVFile(Path.Combine(docPath, "weightsOfOutputLayer.csv"));

            // Формируем выходной слой
            List<Neuron> outputNeurons = new List<Neuron>();
            for (int i = 0; i < outputLayerDimension; i++)
                outputNeurons.Add(new Neuron(ActivationFunctions.Func1, Weights: outputLayerWeights[i]));

            Layer outputLayer = new Layer(LayerType.Output, outputNeurons);
            #endregion

            // Инициализируем нейросеть с помощью слоев (скрытых и выходного)
            Network network = new Network(hiddenLayers, outputLayer);

            // Формируем входной сигнал
            List<double> inputSignals = new List<double> { 0.0, 0.0, 1.0 };

            // Прогоняем входной сигнал через нейросеть и сигналы на выходе
            List<double> outputSignal = network.PropagateForward(inputSignals);
        }
    }
}

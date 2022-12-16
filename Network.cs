using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        public List<Layer> HiddenLayers { get; }
        public Layer OutputLayer { get; }

        /// <summary>
        /// Инициализирует нейросеть с помощью скрытых слоев и выходного слоя
        /// </summary>
        /// <param name="HiddenLayers">список скрытых слоев</param>
        /// <param name="OutputLayer">выходной слой</param>
        public Network(List<Layer> HiddenLayers, Layer OutputLayer)
        {
            this.HiddenLayers = HiddenLayers;
            this.OutputLayer = OutputLayer;
        }        

        /// <summary>
        /// Инициализирует нейросеть с помощью входных параметров
        /// </summary>
        /// <param name="inputLayerDimension">количество входов нейросети</param>
        /// <param name="outputLayerNeuronsCount">количество нейронов на выходном слое</param>
        /// <param name="outputActivationFunction">функция активации у нейронов выходного слоя</param>
        /// <param name="hiddenLayersDimensions">размерности скрытых слоев</param>
        /// <param name="hiddenActivationFunctions">массив функций активаций нейронов скрытых слоев</param>
        public Network(int inputLayerDimension, int outputLayerNeuronsCount, Func<double, double> outputActivationFunction, int[] hiddenLayersDimensions = null, Func<double, double>[] hiddenActivationFunctions = null)
        {
            if (hiddenLayersDimensions.Length != hiddenActivationFunctions.Length)
                throw new Exception("Количество скрытых слоев не равно количеству функций активации для скрытых слоев");

            Random random = new Random();

            // Если есть скрытые слои
            if (hiddenLayersDimensions != null)
            {
                #region Инициализируем весовые коэффициенты на скрытых слоях

                HiddenLayers = new List<Layer>();

                // Сначала инициализируем первый скрытый слой

                // Количество весовых коэффициентов у каждого нейрона первого скрытого слоя равно количеству нейронов входного слоя
                HiddenLayers.Add(new Layer(CreateNeurons(hiddenLayersDimensions[0], inputLayerDimension, -1.0, 1.0, hiddenActivationFunctions[0], random)));

                // Если скрытых слоев больше 1
                if (hiddenLayersDimensions.Length > 1)
                {
                    // Количество весовых коэффициентов на втором и последующих скрытых слоях равно количеству нейронов на предыдущем скрытом слое
                    // Еще раз, первый скрытый слой уже проинициализирован, поэтому начинаем со второго (h = 1)
                    for (int h = 1; h < hiddenLayersDimensions.Length; h++)
                        HiddenLayers.Add(new Layer(CreateNeurons(hiddenLayersDimensions[h], hiddenLayersDimensions[h - 1], -1.0, 1.0, hiddenActivationFunctions[h], random)));
                }

                #endregion
            }

            #region Инициализируем весовые коэффициенты выходного слоя

            // Если есть скрытые слои, то количество весовых коэффицинтов у нейронов выходного слоя равно количеству нейронов последнего скрытого слоя
            // Если скрытых слоев нет, то количество весовых коэффицинтов у нейронов выходного слоя равно количеству нейронов входного слоя

            int outputWeightsCount = hiddenLayersDimensions != null ? hiddenLayersDimensions.Last() : inputLayerDimension;

            OutputLayer = new Layer(CreateNeurons(outputLayerNeuronsCount, outputWeightsCount, -1.0, 1.0, outputActivationFunction, random));

            #endregion
        }

        public List<double> PropagateForward(List<double> functionSignal)
        {
            // Передаем сигнал по скрытым слоям
            foreach (Layer hiddenLayer in HiddenLayers)
                functionSignal = hiddenLayer.ProduceSignals(functionSignal);

            // Возвращаем сигнал от выходного слоя
            return OutputLayer.ProduceSignals(functionSignal);
        }

        /// <summary>
        /// Возвращает список нейронов, инициализированных весовыми коэффициентами
        /// </summary>
        /// <param name="neuronsCount">количество нейронов</param>
        /// <param name="weightsCount">количество весовых коэффициентов в каждом нейроне</param>
        /// <param name="weightsMinValue">левая граница интервала случайных чисел</param>
        /// <param name="weightsMaxValue">правая граница интервала случайных чисел</param>
        /// <param name="activationFunction">функция активации</param>
        /// <param name="random">экземпляр генератора случайных чисел</param>
        /// <returns></returns>
        private List<Neuron> CreateNeurons(int neuronsCount, int weightsCount, double weightsMinValue, double weightsMaxValue, Func<double, double> activationFunction, Random random)
        {
            List<Neuron> neurons = new List<Neuron>();

            for (int i = 0; i < neuronsCount; i++)
            {
                List<double> weights = CreateRandomWeights(weightsCount, weightsMinValue, weightsMaxValue, random);
                neurons.Add(new Neuron(activationFunction, weights, CreateRandomValue(random, weightsMinValue, weightsMaxValue)));
            }

            return neurons;
        }

        /// <summary>
        /// Возвращает случайное число в заданном интервале 
        /// </summary>
        /// <param name="random">экземпляр генератора случайных чисел</param>
        /// <param name="minValue">левая граница интеравала</param>
        /// <param name="maxValue">правая граница интеравала</param>
        /// <returns></returns>
        private double CreateRandomValue(Random random, double minValue, double maxValue)
        {
            return random.NextDouble() * (maxValue - minValue) + minValue;
        }

        /// <summary>
        /// Возвращает список весовых коэффициентов инициализированных случайными значениями
        /// </summary>
        /// <param name="weightsCount">количество весовых коэффициентов</param>
        /// <param name="minValue">левая граница интервала случайных чисел</param>
        /// <param name="maxValue">правая граница интервала случайных чисел</param>
        /// <param name="random">экземпляр генератора случайных чисел</param>
        /// <returns></returns>
        private List<double> CreateRandomWeights(int weightsCount, double minValue, double maxValue, Random random)
        {
            List<double> weights = new List<double>();

            for (int i = 0; i < weightsCount; i++)
                weights.Add(CreateRandomValue(random, minValue, maxValue));

            return weights;
        }

        public void WriteHiddenWeightsToCSVFile(string fileName)
        {
            TextWriter textWriter = new StreamWriter(fileName);

            textWriter.WriteLine(string.Format("{0};{1}", "hiddenLayersDimensions", string.Join(";", HiddenLayers.Select(x => x.Neurons.Count))));

            foreach (Layer hiddenLayer in HiddenLayers)
                foreach (Neuron neuron in hiddenLayer.Neurons)
                    textWriter.WriteLine("{0};{1}", neuron.Bias, string.Join(";", neuron.Weights));

            textWriter.Close();
        }

        public void WriteOutputWeightsToCSVFile(string fileName)
        {
            TextWriter textWriter = new StreamWriter(fileName);

            foreach (Neuron neuron in OutputLayer.Neurons)
                textWriter.WriteLine("{0};{1}", neuron.Bias, string.Join(";", neuron.Weights));

            textWriter.Close();
        }

        public double Train(List<List<double>> trainingSet)
        {
            double totalNetworkErrorEnergySum = 0.0;

            foreach (List<double> functionSignal in trainingSet)
            {
                List<double> desiredResponse = GetDesiredResponse(functionSignal);
                List<double> outputSignal = PropagateForward(functionSignal);
                List<double> errorSignal = GetErrorSignal(desiredResponse, outputSignal);

                double currentErrorEnergy = GetCurrentErrorEnergy(errorSignal);
                totalNetworkErrorEnergySum += currentErrorEnergy;

                PropagateBackward(errorSignal);
            }

            return totalNetworkErrorEnergySum / trainingSet.Count;
        }

        private void PropagateBackward(List<double> errorSignal)
        {
            throw new NotImplementedException();
        }

        private List<double> GetErrorSignal(List<double> desiredResponse, List<double> outputSignal)
        {
            List<double> errorSignal = new List<double>();

            for (int i = 0; i < desiredResponse.Count; i++)
                errorSignal.Add(desiredResponse[i] - outputSignal[i]);

            return errorSignal;
        }

        private double GetCurrentErrorEnergy(List<double> errorSignal)
        {
            double sum = 0.0;

            for (int i = 0; i < errorSignal.Count; i++)
                sum += Math.Pow(errorSignal[i], 2.0);

            return 0.5 * sum;
        }

        private List<double> GetDesiredResponse(List<double> inputSignal)
        {
            throw new NotImplementedException();
        }
    }
}
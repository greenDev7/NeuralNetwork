using MNIST.IO;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        /// <summary>
        /// Скрытые слои
        /// </summary>
        public List<Layer> HiddenLayers { get; }
        /// <summary>
        /// Выходной слой
        /// </summary>
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
        /// <param name="randomMinValue">левая граница для рандомных чисел</param>
        /// <param name="randomMaxValue">правая граница для рандомных чисел</param>
        public Network(int inputLayerDimension, int outputLayerNeuronsCount, Func<double, double> outputActivationFunction, int[] hiddenLayersDimensions = null, 
            Func<double, double>[] hiddenActivationFunctions = null, double randomMinValue = 0.0, double randomMaxValue = 1.0)
        {
            Random random = new Random();

            // Если есть скрытые слои
            if (hiddenLayersDimensions != null)
            {
                //if (hiddenLayersDimensions.Length != hiddenActivationFunctions.Length)
                //    throw new Exception("Количество скрытых слоев не равно количеству функций активации для скрытых слоев");

                #region Инициализируем весовые коэффициенты на скрытых слоях

                HiddenLayers = new List<Layer>();

                // Сначала инициализируем первый скрытый слой

                // Количество весовых коэффициентов у каждого нейрона первого скрытого слоя равно количеству нейронов входного слоя
                HiddenLayers.Add(new Layer(CreateNeurons(hiddenLayersDimensions[0], inputLayerDimension, randomMinValue, randomMaxValue, hiddenActivationFunctions[0], random)));

                // Если скрытых слоев больше 1
                if (hiddenLayersDimensions.Length > 1)
                {
                    // Количество весовых коэффициентов на втором и последующих скрытых слоях равно количеству нейронов на предыдущем скрытом слое
                    // Еще раз, первый скрытый слой уже проинициализирован, поэтому начинаем со второго (h = 1)
                    for (int h = 1; h < hiddenLayersDimensions.Length; h++)
                        HiddenLayers.Add(new Layer(CreateNeurons(hiddenLayersDimensions[h], hiddenLayersDimensions[h - 1], randomMinValue, randomMaxValue, hiddenActivationFunctions[h], random)));
                }

                #endregion
            }

            #region Инициализируем весовые коэффициенты выходного слоя

            // Если есть скрытые слои, то количество весовых коэффицинтов у нейронов выходного слоя равно количеству нейронов последнего скрытого слоя
            // Если скрытых слоев нет, то количество весовых коэффицинтов у нейронов выходного слоя равно количеству входов сети
            int outputWeightsCount = hiddenLayersDimensions != null ? hiddenLayersDimensions.Last() : inputLayerDimension;

            OutputLayer = new Layer(CreateNeurons(outputLayerNeuronsCount, outputWeightsCount, randomMinValue, randomMaxValue, outputActivationFunction, random));

            #endregion
        }
        /// <summary>
        /// Запускает алгоритм прямого распространения сигнала и возвращает ответ от сети
        /// </summary>
        /// <param name="functionSignal">функциональный сигнал (стимул), поступающий на вход нейросети</param>
        /// <returns></returns>
        public List<double> MakePropagateForward(List<double> functionSignal)
        {
            // Если имеются скрытые слои, то передаем сигнал по скрытым слоям
            if (HiddenLayers != null)
                foreach (Layer hiddenLayer in HiddenLayers)
                    functionSignal = SetInputSignalAndInducedLocalFieldAndReturnOutputSignal(hiddenLayer, functionSignal);

            // Возвращаем сигнал от выходного слоя
            return SetInputSignalAndInducedLocalFieldAndReturnOutputSignal(OutputLayer, functionSignal);
        }
        /// <summary>
        /// Задает слою входной сигнал, устанавливает локальные индуцированные поля нейронов и возвращает выходной сигнал
        /// </summary>
        /// <param name="layer">слой</param>
        /// <param name="functionSignal">входной сигнал</param>
        /// <returns>выходной сигнал</returns>
        private List<double> SetInputSignalAndInducedLocalFieldAndReturnOutputSignal(Layer layer, List<double> functionSignal)
        {
            layer.InputSignals = functionSignal;

            foreach (Neuron neuron in layer.Neurons)
                neuron.SetInducedLocalField(functionSignal);
            
            return layer.ProduceSignals();
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
                neurons.Add(new Neuron(activationFunction, weights, CreateRandomValue(random, weightsMinValue, weightsMaxValue, i)));
            }

            return neurons;
        }
        /// <summary>
        /// Возвращает случайное число в заданном интервале 
        /// </summary>
        /// <param name="random">экземпляр генератора случайных чисел</param>
        /// <param name="minValue">левая граница интеравала</param>
        /// <param name="maxValue">правая граница интеравала</param>
        /// <param name="currentIndex">текущий индекс генерируемого значения</param>
        /// <returns></returns>
        private double CreateRandomValue(Random random, double minValue, double maxValue, int currentIndex)
        {
            double randomDouble = random.NextDouble() * (maxValue - minValue) + minValue;

            if (currentIndex % 2 == 0) // Будем чередовать знаки через один
                return -randomDouble;

            return randomDouble;
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
                weights.Add(CreateRandomValue(random, minValue, maxValue, i));

            return weights;
        }
        /// <summary>
        /// Записывает данные по скрытым слоям (количество скрытых слоев, их размерности, пороговые значения, весовые коэффициенты) в csv файл
        /// </summary>
        /// <param name="fileName">имя файла для записи</param>
        public void WriteHiddenWeightsToCSVFile(string fileName)
        {
            if (HiddenLayers == null)
                return;

            TextWriter textWriter = new StreamWriter(fileName);

            textWriter.WriteLine(string.Format("{0};{1}", "hiddenLayersDimensions", string.Join(";", HiddenLayers.Select(x => x.Neurons.Count))));

            foreach (Layer hiddenLayer in HiddenLayers)
                foreach (Neuron neuron in hiddenLayer.Neurons)
                    textWriter.WriteLine("{0};{1}", neuron.Bias, string.Join(";", neuron.Weights));

            textWriter.Close();
        }        
        /// <summary>
        /// Записывает весовые коэффициенты выходного слоя сети в csv файл
        /// </summary>
        /// <param name="fileName">имя файла для записи</param>
        public void WriteOutputWeightsToCSVFile(string fileName)
        {
            TextWriter textWriter = new StreamWriter(fileName);

            foreach (Neuron neuron in OutputLayer.Neurons)
                textWriter.WriteLine("{0};{1}", neuron.Bias, string.Join(";", neuron.Weights));

            textWriter.Close();
        }        
        /// <summary>
        /// Запускает алгоритм обучения нейронной сети
        /// </summary>
        /// <param name="imagesFileName">путь к бинарному файлу MNIST с изображениями</param>
        /// <param name="labelsFileName">путь к бинарному файлу MNIST с метками (наименованиями цифр)</param>
        /// <param name="learningRateParameter">параметр скорости обучения</param>
        /// <param name="numberOfEpochs">количество эпох</param>
        /// <returns>Массив значений общей энергии ошибки</returns>
        public List<double> Train(string imagesFileName, string labelsFileName, double learningRateParameter, int numberOfEpochs)
        {
            List<double> currentErrorList = new List<double>();

            for (int e = 0; e < numberOfEpochs; e++)
            {
                IEnumerable<TestCase> testCases = FileReaderMNIST.LoadImagesAndLables(labelsFileName, imagesFileName);

                foreach (TestCase test in testCases)
                {
                    List<double> functionSignal = ImageHelper.ConvertImageToFunctionSignal(test.Image);
                    List<double> desiredResponse = GetDesiredResponse(test.Label);

                    List<double> outputSignal = MakePropagateForward(functionSignal);

                    List<double> errorSignal = GetErrorSignal(desiredResponse, outputSignal);
                    //double currentErrorEnergy = GetCurrentErrorEnergy(errorSignal);
                    //currentErrorList.Add(currentErrorEnergy);

                    MakePropagateBackward(errorSignal, learningRateParameter);
                }

                Console.WriteLine("epoch " + e.ToString() + " finished");
            }

            return currentErrorList;
        }
        /// <summary>
        /// Тестовый метод. Использовался мною для отладки метода Train()
        /// </summary>
        /// <param name="totalErrorEnergy">Общая энергия ошибки сети</param>
        /// <param name="learningRateParameter">параметр скорости обучения</param>
        /// <returns>Массив значений общей энергии ошибки</returns>
        public List<double> TestTrain(out double totalErrorEnergy, double learningRateParameter)
        {
            TestTrain test = new TestTrain();

            double totalNetworkErrorEnergySum = 0.0;

            List<double> currentErrorList = new List<double>();

            foreach ((List<double>, List<double>) ex in test.TrainingSet)
            {
                List<double> outputSignal = MakePropagateForward(ex.Item1);

                List<double> errorSignal = GetErrorSignal(ex.Item2, outputSignal);
                double currentErrorEnergy = GetCurrentErrorEnergy(errorSignal);
                currentErrorList.Add(currentErrorEnergy);
                totalNetworkErrorEnergySum += currentErrorEnergy;

                MakePropagateBackward(errorSignal, learningRateParameter);
            }

            totalErrorEnergy = totalNetworkErrorEnergySum / test.TrainingSet.Count;

            return currentErrorList;
        }
        /// <summary>
        /// Запускает алгоритм обратного распространения ошибки
        /// </summary>
        /// <param name="errorSignal">сигнал ошибки</param>
        /// <param name="learningRateParameter">параметр скорости обучения</param>
        private void MakePropagateBackward(List<double> errorSignal, double learningRateParameter)
        {
            OutputLayer.CalculateAndSetLocalGradients(errorSignal); // Вычисляем локальные градиенты для выходного слоя
            OutputLayer.AdjustWeights(learningRateParameter); // Корректируем весовые коэффициенты

            // Если скрытых слоев нет, то заканчиваем процесс
            if (HiddenLayers == null)
                return;

            Layer previousLayer = OutputLayer; 

            // Вычисляем локальные градиенты для скрытых слоев
            for (int i = HiddenLayers.Count - 1; i >= 0; i--)
            {
                HiddenLayers[i].CalculateAndSetLocalGradients(previousLayer);
                HiddenLayers[i].AdjustWeights(learningRateParameter);
                previousLayer = HiddenLayers[i];
            }           
        }
        /// <summary>
        /// Возвращает сигнал ошибки сети
        /// </summary>
        /// <param name="desiredResponse">желаемый отклик сети</param>
        /// <param name="outputSignal">действительный отклик сети (функциональный сигнал, генерируемый на выходе работы сети)</param>
        /// <returns>Сигнал ошибки</returns>
        private List<double> GetErrorSignal(List<double> desiredResponse, List<double> outputSignal)
        {
            List<double> errorSignal = new List<double>();

            for (int i = 0; i < desiredResponse.Count; i++)
                errorSignal.Add(desiredResponse[i] - outputSignal[i]);

            return errorSignal;
        }
        /// <summary>
        /// Возвращает текущую энергию ошибки сети
        /// </summary>
        /// <param name="errorSignal">сигнал ошибки сети</param>
        /// <returns></returns>
        private double GetCurrentErrorEnergy(List<double> errorSignal)
        {
            double sum = 0.0;

            for (int i = 0; i < errorSignal.Count; i++)
                sum += Math.Pow(errorSignal[i], 2.0);

            return 0.5 * sum;
        }
        /// <summary>
        /// Возвращает желаемый отклик нейросети
        /// </summary>
        /// <param name="digit">цифра (от 0 до 9)</param>
        /// <returns></returns>
        private List<double> GetDesiredResponse(int digit)
        {
            List<double> desiredResponse = new List<double>() { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

            desiredResponse[digit] = 1.0;

            return desiredResponse;
        }
    }
}
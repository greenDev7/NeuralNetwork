# NeuralNetwork

Реализация классического алгоритма обратного распространения ошибки с помощью последовательного (стохастического) режима обучения для тренировки многослойного персептрона распознаванию рукописных цифр

![](https://github.com/greenDev7/NeuralNetwork/blob/master/DigitRecognition.gif)

[Digit Recognition Canvas (песочница)](https://codesandbox.io/s/winter-dew-u26xeb)

[Digit Recognition Canvas (исходники)](https://github.com/greenDev7/DigitRecognitionCanvas)

[Описание алгоритма (Хабр)](https://habr.com/ru/post/708928/)

## Form1.cs


``` csharp
private void learnButton_Click(object sender, EventArgs e)
        {
            string myDocumentFolder = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);

            string trainingImagesPath = Path.Combine(myDocumentFolder, "train-images-idx3-ubyte");
            string trainingLabelsPath = Path.Combine(myDocumentFolder, "train-labels-idx1-ubyte");

            string testImagesPath = Path.Combine(myDocumentFolder, "t10k-images-idx3-ubyte");
            string testLabelsPath = Path.Combine(myDocumentFolder, "t10k-labels-idx1-ubyte");

            #region Блок для инициализации нейросети с помощью весовых коэффициентов из файлов csv

            //// Считываем весовые коэффициенты из файлов
            // List<Layer> hiddenLayers = InitializeHiddenLayersWeightsFromCSVFile(Path.Combine(myDocumentFolder, "adjustedHiddenLayerWeights_acc9572_16.csv"));
            // Layer outputLayer = InitializeOutputLayerWeightsFromCSVFile(Path.Combine(myDocumentFolder, "adjustedOutputLayerWeights_acc9572_16.csv"));
            ////Инициализируем нейросеть
            //Network network = new Network(hiddenLayers, outputLayer);
            
            #endregion


            #region Блок для инициализация нейросети рандомными значениями и ее обучение

            // Инициализируем нейросеть с помощью заданных параметров

            int hiddenLayersCount = 1;  // Задаем количество скрытых слоев
            int[] hiddenLayersDimensions = new int[hiddenLayersCount]; // Массив для хранения количества нейронов на каждом скрытом слое
            Func<double, double>[] hiddenActivationFunctions = new Func<double, double>[hiddenLayersCount]; // Массив для хранения функций активации на каждом скрытом слое

            hiddenLayersDimensions[0] = 80; // У нас один скрытый слой на котором 80 нейронов
            hiddenActivationFunctions[0] = ActivationFunctions.SigmoidFunction; // И для всех нейронов этого скрытого слоя используется сигмоидальная функция активации

            // 784 входа - это размер массива полученного из изображения (28 * 28 пикселей)
            Network network = new Network(784, 10, ActivationFunctions.SigmoidFunction, hiddenLayersDimensions, hiddenActivationFunctions);
            network.Train(trainingImagesPath, trainingLabelsPath, 0.2, 16); // Запускаем обучение

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
                    // Получим это изображение
                    Bitmap bitmap = ImageHelper.CreateBitmapFromMnistImage(test.Image);                    
                    // И сохраним в папку IncorrectPredictions
                    bitmap.Save(Path.Combine(myDocumentFolder, "IncorrectPredictions", $"{incorrectPredictionsCount}_{test.Label}_{predictedDigit}.png"));
                }
            }

            double accuracy = 100.0 - (incorrectPredictionsCount / 100.0); // Вычисляем точность (%)
            #endregion


            // Записываем скорректированные весовые коэффициенты в файлы
            network.WriteHiddenWeightsToCSVFile(Path.Combine(myDocumentFolder, $"adjustedHiddenLayerWeights_acc{accuracy.ToString().Replace(",", string.Empty)}.csv"));
            network.WriteOutputWeightsToCSVFile(Path.Combine(myDocumentFolder, $"adjustedOutputLayerWeights_acc{accuracy.ToString().Replace(",", string.Empty)}.csv"));
            // и в JSON файлы
            network.WriteHiddenWeightsToJsonFile(Path.Combine(myDocumentFolder, $"adjustedHiddenLayerWeights_acc{accuracy.ToString().Replace(",", string.Empty)}.json"));
            network.WriteOutputWeightsToJsonFile(Path.Combine(myDocumentFolder, $"adjustedOutputLayerWeights_acc{accuracy.ToString().Replace(",", string.Empty)}.json"));
        }
```
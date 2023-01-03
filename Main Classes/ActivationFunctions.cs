using System;

namespace NeuralNetwork
{
    public static class ActivationFunctions
    {
        /// <summary>
        /// Параметр для сигмоидальной функции активации
        /// </summary>
        private const double a = 1.0;        

        /// <summary>
        /// Пороговая функция активации (использовалась для тестирования для решения проблемы XOR)
        /// </summary>
        /// <param name="x">аргумент функции</param>
        /// <returns></returns>
        public static double ThresholdFunction(double x)
        {
            return x >= 0.0 ? 1.0 : 0.0;
        }
        /// <summary>
        /// Возвращает значение сигмоидальной функции активации
        /// </summary>
        /// <param name="x">аргумент функции</param>
        /// <returns></returns>
        public static double SigmoidFunction(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -a * x));
        }
        /// <summary>
        /// Возвращает значение производной сигмоидальной функции активации (для алгоритма обратного распространения)
        /// </summary>
        /// <param name="x">аргумент функции</param>
        /// <returns></returns>
        public static double SigmoidFunctionsDerivative(double x)
        {
            double factor = a * Math.Pow(Math.E, -a * x);

            return factor * Math.Pow(SigmoidFunction(x), 2.0);
        }
    }
}

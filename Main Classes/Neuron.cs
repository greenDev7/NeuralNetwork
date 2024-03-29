﻿using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Neuron
    {
        /// <summary>
        /// Весовые коэффициенты (синаптические связи)
        /// </summary>
        public List<double> Weights { get; }
        /// <summary>
        /// Пороговое значение
        /// </summary>
        public double Bias { get; private set; }
        /// <summary>
        /// Функция активации
        /// </summary>
        private Func<double, double> ActivationFunction { get; }
        /// <summary>
        /// Индуцированное локальное поле
        /// </summary>
        public double InducedLocalField { get; private set; }
        /// <summary>
        /// Локальный градиент
        /// </summary>
        public double LocalGradient { get; private set; }

        /// <summary>
        /// Инициализирует нейрон
        /// </summary>
        /// <param name="ActivationFunction">функция активации</param>
        /// <param name="Weights">список весовых коэффициентов</param>
        /// <param name="Bias">пороговое значение</param>
        public Neuron(Func<double, double> ActivationFunction, List<double> Weights = null, double Bias = 0.0)
        {
            this.ActivationFunction = ActivationFunction;            
            this.Weights = Weights;
            this.Bias = Bias;
            this.InducedLocalField = 0.0;
        }        
        
        /// <summary>
        /// Устанавливает локальное индуцированное поля для данного нейрона
        /// </summary>
        /// <param name="inputSignal">входной сигнал</param>
        public void SetInducedLocalField(List<double> inputSignal)
        {
            //if (inputSignal.Count != Weights.Count)
            //    throw new ArgumentOutOfRangeException("inputSignals", "Ошибка при вычислении индуцированного локального поля: количество входных сигналов не равно количеству весовых коэффициентов");

            InducedLocalField = 0.0;

            for (int i = 0; i < inputSignal.Count; i++)
                InducedLocalField += inputSignal[i] * Weights[i];

            InducedLocalField += Bias;
        }
        /// <summary>
        /// Устанавливет значение локального градиента
        /// </summary>
        /// <param name="localGradient">локальный градиент</param>
        internal void SetLocalGradient(double localGradient) => LocalGradient = localGradient;
        /// <summary>
        /// Корректирует весовые коэффициенты
        /// </summary>
        /// <param name="learningRateParameter">параметр скорости обучения</param>
        /// <param name="inputSignals">входной сигнал нейрона, заданный при прямом проходе</param>
        internal void AdjustWeightsAndBias(double learningRateParameter, List<double> inputSignals)
        {
            for (int i = 0; i < Weights.Count; i++)
            {
                Weights[i] += learningRateParameter * LocalGradient * inputSignals[i]; // Корректируем весовые коэффициенты
                // Bias += learningRateParameter * LocalGradient; // Корректируем пороговое значение (inputSignal для него равен 1)
            }
        }
        /// <summary>
        /// Возвращает функциональный сигнал от данного нейрона
        /// </summary>
        /// <returns></returns>
        public double GetActivationPotential()
        {
            //if (InducedLocalField == null)
            //    throw new ArgumentNullException("InducedLocalField", "В функцию активации передан аргумент равный null");
            //else
            //    return ActivationFunction((double)InducedLocalField);

            return ActivationFunction(InducedLocalField);
        }        
    }
}
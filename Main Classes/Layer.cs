using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        /// <summary>
        /// Нейроны
        /// </summary>
        public List<Neuron> Neurons { get; }
        /// <summary>
        /// Входной сигнал
        /// </summary>
        public List<double> InputSignals { get; set; }

        public Layer(List<Neuron> Neurons)
        {
            this.Neurons = Neurons;            
        }

        /// <summary>
        /// Возвращает функциональный сигнал от нейронов данного слоя
        /// </summary>
        /// <returns></returns>
        internal List<double> ProduceSignals()
        {
            List<double> currentSignals = new List<double>();

            foreach (Neuron neuron in Neurons)
                currentSignals.Add(neuron.GetActivationPotential());
       
            return currentSignals;
        }
        /// <summary>
        /// Вычисляет и устанавливает нейронам ВЫХОДНОГО слоя локальные градиенты
        /// </summary>
        /// <param name="errorSignal">сигнал ошибки</param>
        internal void CalculateAndSetLocalGradients(List<double> errorSignal)
        {
            for (int i = 0; i < Neurons.Count; i++)
                Neurons[i].SetLocalGradient(errorSignal[i] * ActivationFunctions.SigmoidFunctionsDerivative(Neurons[i].InducedLocalField));
        }
        /// <summary>
        /// Вычисляет и устанавливает нейронам СКРЫТОГО слоя локальные градиенты на основе предыдущего слоя в алгоритме обратного распространения
        /// </summary>
        /// <param name="previousLayer">предыдущий слой (расположенный правее текущего)</param>
        internal void CalculateAndSetLocalGradients(Layer previousLayer)
        {
            for (int i = 0; i < Neurons.Count; i++)
            {
                List<double> associatedWeights = GetAssociatedWeights(previousLayer, i);

                double innerSum = GetInnerSum(associatedWeights, previousLayer);

                Neurons[i].SetLocalGradient(innerSum * ActivationFunctions.SigmoidFunctionsDerivative(Neurons[i].InducedLocalField));
            }
        }
        /// <summary>
        /// Возвращает локальное скалярное произведение (см. Хайкин, стр. 231, формула 4.24)
        /// </summary>
        /// <param name="associatedWeights">синаптические связи (весовые коэффициенты, связывающие нейрон текущего слоя и нейроны предыдущего)</param>
        /// <param name="previousLayer">предыдущий слой нейронов</param>
        /// <returns></returns>
        private double GetInnerSum(List<double> associatedWeights, Layer previousLayer)
        {
            double innerSum = 0.0;

            for (int i = 0; i < associatedWeights.Count; i++)
                innerSum += associatedWeights[i] * previousLayer.Neurons[i].LocalGradient;

            return innerSum;
        }
        /// <summary>
        /// Возвращает весовые коэффициенты, связывающие нейрон текущего слоя и нейроны предыдущего слоя
        /// </summary>
        /// <param name="previousLayer">предыдущий слой</param>
        /// <param name="neuronPositionInCurrentLayer">номер нейрона в текущем слое</param>
        /// <returns></returns>
        private List<double> GetAssociatedWeights(Layer previousLayer, int neuronPositionInCurrentLayer)
        {
            List<double> associatedWeights = new List<double>();

            foreach (Neuron neuron in previousLayer.Neurons)
                associatedWeights.Add(neuron.Weights[neuronPositionInCurrentLayer]);

            return associatedWeights;
        }
        /// <summary>
        /// Корректирует весовые коэффициенты нейронов данного слоя
        /// </summary>
        /// <param name="learningRateParameter">параметр скорости обучения</param>
        internal void AdjustWeightsAndBias(double learningRateParameter)
        {
            for (int i = 0; i < Neurons.Count; i++)
                Neurons[i].AdjustWeightsAndBias(learningRateParameter, InputSignals);
        }
    }
}
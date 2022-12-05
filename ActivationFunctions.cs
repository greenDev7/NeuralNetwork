using System;

namespace NeuralNetwork
{
    public static class ActivationFunctions
    {
        public static double ThresholdFunction(double x)
        {
            return x >= 0.0 ? 1.0 : 0.0;
        }

        public static double LogisticFunction(double x, double a = 1.0)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -a * x));
        }
    }
}

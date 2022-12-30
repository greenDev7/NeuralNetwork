using System;

namespace NeuralNetwork
{
    public static class ActivationFunctions
    {
        private const double a = 1.0;        

        public static double ThresholdFunction(double x)
        {
            return x >= 0.0 ? 1.0 : 0.0;
        }

        public static double LogisticFunction(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -a * x));
        }

        public static double LogisticFunctionsDerivative(double x)
        {
            double factor = a * Math.Pow(Math.E, -a * x);

            return factor * Math.Pow(LogisticFunction(x), 2.0);
        }
    }
}

using System.Collections.Generic;

namespace NeuralNetwork
{
    public class TestTrain
    {
        public List<(List<double>, List<double>)> TrainingSet { get; set; }

        public TestTrain()
        {
            TrainingSet = new List<(List<double>, List<double>)>();

            TrainingSet.Add((new List<double>() { 0.0, 0.0, 1.0 }, new List<double>() { 0.0, 1.0, 0.0 }));
            //TrainingSet.Add((new List<double>() { 0.0, 1.0, 0.0 }, new List<double>() { 1.0, 0.0, 0.0 }));
            //TrainingSet.Add((new List<double>() { 1.0, 0.0, 0.0 }, new List<double>() { 0.0, 0.0, 1.0 }));
        }
    }
}

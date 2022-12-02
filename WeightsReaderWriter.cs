using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public static class WeightsReaderWriter
    {
        public static List<List<List<double>>> ReadHiddenLayersWeightsFromCSVFile(int hiddenLayersCount, int[] hiddenLayerDimensions, string weightsOfHiddenLayerFileName)
        {
            List<List<List<double>>> hiddenLayersWeights = new List<List<List<double>>>();

            string[] lines = File.ReadAllLines(weightsOfHiddenLayerFileName);

            int startPosition = 0;
            for (int i = 0; i < hiddenLayersCount; i++)
            {
                List<List<double>> currentLayerWeights = new List<List<double>>();
               
                for (int j = startPosition; j < startPosition + hiddenLayerDimensions[i]; j++)
                {
                    List<double> doubleList = lines[j].Split(new string[] { ";" }, StringSplitOptions.RemoveEmptyEntries).Select(x => Convert.ToDouble(x)).ToList();
                    currentLayerWeights.Add(doubleList);
                }

                startPosition += hiddenLayerDimensions[i];
                hiddenLayersWeights.Add(currentLayerWeights);                
            }

            return hiddenLayersWeights;
        }

        public static List<List<double>> ReadOutputLayerWeightsFromCSVFile(string outputWeightsFile)
        {
            List<List<double>> outputLayerWeights = new List<List<double>>();

            string[] lines = File.ReadAllLines(outputWeightsFile);

            foreach (string line in lines)
                outputLayerWeights.Add(line.Split(';').Select(x => Convert.ToDouble(x)).ToList());

            return outputLayerWeights;
        }
    }
}

﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    public static class WeightsReader
    {
        public static List<List<List<double>>> ReadHiddenLayersWeightsFromCSVFile(string fileName)
        {
            List<List<List<double>>> hiddenLayersWeights = new List<List<List<double>>>();

            string[] lines = File.ReadAllLines(fileName);

            // Считываем размерности скрытых слоев
            List<int> hiddenLayerDimensions = lines[0].Split(';').Skip(1).Select(x => Convert.ToInt32(x)).ToList();

            int startPosition = 1;
            for (int i = 0; i < hiddenLayerDimensions.Count; i++)
            {
                List<List<double>> currentLayerWeights = new List<List<double>>();
               
                for (int j = startPosition; j < startPosition + hiddenLayerDimensions[i]; j++)
                {
                    List<double> doubleList = lines[j].Split(';').Select(x => Convert.ToDouble(x)).ToList();
                    currentLayerWeights.Add(doubleList);
                }

                startPosition += hiddenLayerDimensions[i];
                hiddenLayersWeights.Add(currentLayerWeights);                
            }

            return hiddenLayersWeights;
        }
        public static List<List<double>> ReadOutputLayerWeightsFromCSVFile(string fileName)
        {
            List<List<double>> outputLayerWeights = new List<List<double>>();

            string[] lines = File.ReadAllLines(fileName);         

            foreach (string line in lines)
                outputLayerWeights.Add(line.Split(';').Select(x => Convert.ToDouble(x)).ToList());

            return outputLayerWeights;
        }
    }
}
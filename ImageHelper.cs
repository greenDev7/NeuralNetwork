using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace NeuralNetwork
{
    public static class ImageHelper
    {
        public static List<double> ConvertImageToFunctionSignal(string fileName)
        {
            List<double> functionSignal = new List<double>();

            Bitmap img = new Bitmap(fileName);            

            for (int i = 0; i < img.Width; i++)
                for (int j = 0; j < img.Height; j++)
                {
                    // Если пиксель белый
                    if (img.GetPixel(i, j).ToArgb() == -1)
                        functionSignal.Add(0.0);
                    else
                        functionSignal.Add(1.0);
                }

            return functionSignal;
        }

        public static List<List<double>> ConvertImageToPixelMatrix(string fileName)
        {
            Bitmap img = new Bitmap(fileName);

            List<List<double>> pixelMatrix = new List<List<double>>();

            for (int i = 0; i < img.Height; i++)
            {
                List<double> pixelLine = new List<double>();

                for (int j = 0; j < img.Width; j++)
                {
                    // Если пиксель белый
                    if (img.GetPixel(j, i).ToArgb() == -1)
                        pixelLine.Add(0.0);
                    else
                        pixelLine.Add(1.0);
                }

                pixelMatrix.Add(pixelLine);
            }

            return pixelMatrix;
        }

        public static void WritePixelMatrixToCSVFile(List<List<double>> pixelMatrix, string fileName)
        {
            using (StreamWriter streamWriter = new StreamWriter(fileName))
            {
                foreach (List<double> pixelLine in pixelMatrix)
                    streamWriter.WriteLine(string.Join(";", pixelLine));
            }
        }
    }
}

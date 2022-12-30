using MNIST.IO;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

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
        public static List<List<int>> ConvertBytesToPixelMatrix(byte[,] image)
        {
            List<List<int>> pixelMatrix = new List<List<int>>();

            for (int i = 0; i < image.GetLength(0); i++)
            {
                List<int> pixelLine = new List<int>();

                for (int j = 0; j < image.GetLength(1); j++)
                {
                    pixelLine.Add(image[i, j]);
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
        public static Bitmap CreateBitmapFromMnistImage(byte[,] mnistImage)
        {
            int pixelsCount = 28;

            Bitmap img = new Bitmap(pixelsCount, pixelsCount);

            for (int i = 0; i < pixelsCount; i++)
            {
                for (int j = 0; j < pixelsCount; j++)
                {
                    int colorComponent = 255 - mnistImage[i, j];
                    Color newColor = Color.FromArgb(colorComponent, colorComponent, colorComponent);
                    img.SetPixel(j, i, newColor);
                }
            }

            return img;
        }
        public static List<(int, string)> GetRandomImagesPaths(string path)
        {
            List<(int, string)> imagesPaths = new List<(int, string)>();

            string[] catalogsWithDigits = Directory.GetDirectories(path);

            for (int i = 0; i < catalogsWithDigits.Length; i++)
            {
                string[] images = Directory.GetFiles(catalogsWithDigits[i]);

                for (int j = 0; j < images.Length; j++)
                    imagesPaths.Add((i, images[j]));
            }

            Random rnd = new Random();
            return imagesPaths.OrderBy(x => rnd.Next()).ToList();
        }
        public static void CreateImagesFromMnistFile(string path, string imagesFileName, string labelsFileName)
        {
            Directory.CreateDirectory(path);

            for (int i = 0; i < 10; i++)
                Directory.CreateDirectory(Path.Combine(path, i.ToString()));


            IEnumerable<TestCase> testCases = FileReaderMNIST.LoadImagesAndLables(labelsFileName, imagesFileName);

            List<int> captions = new List<int>() { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

            foreach (TestCase test in testCases)
            {
                Bitmap bitmap = CreateBitmapFromMnistImage(test.Image);
                bitmap.Save(Path.Combine(path, test.Label.ToString(), $"{captions[test.Label]++}.png"));
            }
        }
    }
}

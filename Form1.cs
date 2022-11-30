using System.Drawing;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            //string docPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);

            //Graphics graphics = pictureBox1.CreateGraphics();

            Bitmap bitmap = new Bitmap(500, 500);

            Graphics graphics = Graphics.FromImage(bitmap);

    

            Pen pen = new Pen(Color.Red, 3f);
            graphics.DrawLine(pen, 0, 0, 0, 400);

            pictureBox1.Image = bitmap;
            //pictureBox1.Scale(new SizeF());

            //pictureBox1.Load(Path.Combine(docPath, "2_3.png"));

            //Graphics myCanvas = pictureBox1.CreateGraphics();

            //pictureBox1.Size = new Size(50, 50);

            

            //myCanvas.DrawLine(new Pen(Color.Red, 3.0f), 20, 20, 30, 30);


           
            //int width = Convert.ToInt32(drawImage.Width);
            //int height = Convert.ToInt32(drawImage.Height);

            //int width = 50;
            //int height = 50;

            //using (Bitmap bmp = new Bitmap(width, height))
            //{
            //    pictureBox1.DrawToBitmap(bmp, new Rectangle(new Point(0, 0), new Size(width, height)));
            //    bmp.Save(Path.Combine(docPath, "22563.png"), ImageFormat.Png);
            //}
        }
    }
}

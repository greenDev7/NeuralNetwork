
namespace NeuralNetwork
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.learnButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // learnButton
            // 
            this.learnButton.Location = new System.Drawing.Point(12, 12);
            this.learnButton.Name = "learnButton";
            this.learnButton.Size = new System.Drawing.Size(412, 67);
            this.learnButton.TabIndex = 0;
            this.learnButton.Text = "Обучить нейронную сеть";
            this.learnButton.UseVisualStyleBackColor = true;
            this.learnButton.Click += new System.EventHandler(this.learnButton_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(436, 91);
            this.Controls.Add(this.learnButton);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "Form1";
            this.Text = "Распознавание рукописных цифр";
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button learnButton;
    }
}


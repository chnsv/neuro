using System;
using System.Drawing;
using System.Linq;
using System.IO;
using System.Windows.Forms;
using OneProject.NeuroNet;

namespace OneProject
{
    public partial class Form1 : Form
    {
        private NeuroNet.NetWork net = new NeuroNet.NetWork();

        private double[] A = new double[15] { 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d };
        public Form1()
        {
            InitializeComponent();
            A = new double[15];
            net = new NetWork();
        }

        private void changeButton(Button b, int ind)
        {
            if (b.BackColor == Color.Black)
            {
                b.BackColor = Color.White;
                A[ind] = 1d;
            }
            else if (b.BackColor == Color.White)
            {
                b.BackColor = Color.Black;
                A[ind] = 0d;
            }
        }
        private void button1_Click(object sender, EventArgs e)
        {
            changeButton(button1, 0);
        }
        private void button2_Click(object sender, EventArgs e)
        {
            changeButton(button2, 0);
        }
        private void button3_Click(object sender, EventArgs e)
        {
            changeButton(button3, 0);
        }
        private void button4_Click(object sender, EventArgs e)
        {
            changeButton(button4, 0);
        }
        private void button5_Click(object sender, EventArgs e)
        {
            changeButton(button5, 0);
        }
        private void button6_Click(object sender, EventArgs e)
        {
            changeButton(button6, 0);
        }
        private void button7_Click(object sender, EventArgs e)
        {
            changeButton(button7, 0);
        }
        private void button8_Click(object sender, EventArgs e)
        {
            changeButton(button8, 0);
        }
        private void button9_Click(object sender, EventArgs e)
        {
            changeButton(button9, 0);
        }
        private void button10_Click(object sender, EventArgs e)
        {
            changeButton(button10, 0);
        }
        private void button11_Click(object sender, EventArgs e)
        {
            changeButton(button11, 0);
        }
        private void button12_Click(object sender, EventArgs e)
        {
            changeButton(button12, 0);
        }
        private void button13_Click(object sender, EventArgs e)
        {
            changeButton(button13, 0);
        }
        private void button14_Click(object sender, EventArgs e)
        {
            changeButton(button14, 0);
        }
        private void button15_Click(object sender, EventArgs e)
        {
            changeButton(button15, 0);
        }
        private void buttonSaveTrainSample_Click(object sender, EventArgs e)
        {
            buttonSaveTrainSample_Click(numericUpDownTrue.Value, A);
        }
        private void buttonSaveTrainSample_Click(decimal vale, double[] input)
        {
            string pathDir;// каталог с файлом обучающей выборки
            string nameFileTrain;// имя файла обучающей выборки
            pathDir = AppDomain.CurrentDomain.BaseDirectory;
            nameFileTrain = pathDir + "Train.txt";
            string[] tmpStr = new string[1];
            tmpStr[0] = vale.ToString();
            for (int i = 0; i < 15; i++)
            {
                tmpStr[0] += input[i].ToString();
            }
            File.AppendAllLines(nameFileTrain, tmpStr);
        }
        private void buttonSaveTestSample_Click(object sender, EventArgs e)
        {
            buttonSaveTestSample_Click(numericUpDownTrue.Value, A);
        }
        private void buttonSaveTestSample_Click(decimal vale, double[] input)
        {
            string pathDir;// каталог с файлом обучающей выборки
            string nameFileTrain;// имя файла обучающей выборки
            pathDir = AppDomain.CurrentDomain.BaseDirectory;
            nameFileTrain = pathDir + "Test.txt";
            string[] tmpStr = new string[1];
            tmpStr[0] = vale.ToString();
            for (int i = 0; i < 15; i++)
            {
                tmpStr[0] += input[i].ToString();
            }
            File.AppendAllLines(nameFileTrain, tmpStr);
        }

        private void buttonRaspoznat_Click(object sender, EventArgs e)
        {
            net.ForwardPass(net, A);
            labelOtvet.Text = net.fact.ToList().IndexOf(net.fact.Max()).ToString();
        }

        private void buttonTrain_Click(object sender, EventArgs e) //обработчик кнопки обучить 
        {
            net.Train(net);
        }
    }
}

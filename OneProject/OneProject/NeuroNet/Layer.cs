using System;
using System.IO;
using System.Windows.Forms;

namespace OneProject.NeuroNet
{
    abstract class Layer
    {
        protected string name_Layer;
        string pathDirWeights;
        string pathFileWeights;
        protected int numofneurons;
        protected int numofprevneurons;
        protected const double learningrate = 0.005d;
        protected const double momentum = 0.05d;
        protected double[,] lastdeltaweights;
        Neuron[] _neurons;

        public Neuron[] Neurons { get => _neurons; set => _neurons = value; }

        public double[] Data
        {
            set
            {
                for(int i =0;i< _neurons.Length; i++)
                {
                    Neurons[i].Inputs = value;
                    Neurons[i].Activator(Neurons[i].Inputs, Neurons[i].Weights);
                }
            }
        }
        protected Layer(int non, int nopn, NeuronType nt, string nm_Layer)
        {
            numofneurons = non;
            numofprevneurons = nopn;
            Neurons = new Neuron[non];
            name_Layer = nm_Layer;
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";
            double[,] Weights;
            if (File.Exists(pathFileWeights))
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights);
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }
            lastdeltaweights = new double[non, nopn + 1];
            for(int i = 0; i < non; i++)
            {
                double[] tmp_weights = new double[nopn + 1];
                for(int j = 0; j < nopn + 1; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(tmp_weights, nt);
            }
        }
        public double[,] WeightInitialize(MemoryMode mm, string path)// метод работы с массивом синаптических весов
        {
            char[] delim = new char[] { ';', ' ' };
            string tmpStr;
            string[] tmpStrWeights;
            double[,] weights = new double[numofneurons, numofprevneurons + 1];
            switch (mm)
            {
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path);
                    string[] memory_element;
                    for(int i = 0; i < numofneurons; i++)
                    {
                        memory_element = tmpStrWeights[i].Split(delim);
                        for(int j = 0; j < numofprevneurons; j++)
                        {
                            weights[i, j] = double.Parse(memory_element[j].Replace(',', '.'), System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;
                case MemoryMode.INIT:
                    Random rand = new Random();
                    for(int i = 0; i < numofneurons; i++)
                    {
                        for(int j = 0; j < numofprevneurons; j++)
                        {
                            weights[i, j] = rand.NextDouble() * 2 - 1;
                            //не сохраняет веса
                        }
                    }
                    break;
                case MemoryMode.SET:
                    using (StreamWriter writer = new StreamWriter(path))
                    {
                        for(int i = 0; i < numofneurons; i++)
                        {
                            for( int j = 0; j < numofprevneurons + 1; j++)
                            {
                                writer.Write(weights[i, j].ToString(System.Globalization.CultureInfo.InvariantCulture));
                                if(j < numofprevneurons)
                                {
                                    writer.Write(";");
                                }
                            }
                            writer.WriteLine();
                        }
                    }
                    break;
            }
            return weights;
        }
        abstract public void Recognize(NetWork net, Layer nextLayer);
        abstract public double[] BackwardPass(double[] stuff);
    }
}

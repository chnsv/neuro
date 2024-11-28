using System;
using System.Diagnostics.Tracing;

namespace OneProject.NeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string type) : base(non, nopn, nt, type) { }
        public override void Recognize(NetWork net, Layer nextLayer)
        {
            double[] hidden_out = new double[Neurons.Length];
            for( int i = 0; i < Neurons.Length; i++)
            {
                hidden_out[i] = Neurons[i].Output;
            }
            nextLayer.Data = hidden_out;
        }
        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];
            // прописать код обучения нейросети
            for(int j = 0; j < numofprevneurons; j++) //цикл вычисления градиентной суммы
            {
                double sum = 0;
                for(int k = 0; k < numofneurons; k++)
                {
                    sum += Neurons[k].Weights[j] * Neurons[k].Derivative * gr_sums[k]; //через градиентные суммы и производную
                }
                gr_sum[j] = sum;
            }
            for(int  i = 0; i < numofneurons; i++) //цикл коррекции синаптических весов
            {
                for(int n = 0;n < numofprevneurons +1; n++)
                {
                    double deltaw;
                    if (n == 0) //если порог
                        deltaw = momentum * lastdeltaweights[i, 0] + learningrate * Neurons[i].Derivative * gr_sums[i];
                    else
                        deltaw = momentum * lastdeltaweights[i, n] + learningrate * Neurons[i].Inputs[n - 1] * Neurons[i].Derivative * gr_sums[i];
                    lastdeltaweights[i, n] = deltaw;
                    Neurons[i].Weights[n] += deltaw; // коррекция весов
                }
            }
            return gr_sum;
        }
    }
}
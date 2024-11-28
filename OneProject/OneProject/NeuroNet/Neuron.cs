using System.Net.Http.Headers;
using System.Runtime.Serialization.Formatters;
using static System.Math;

namespace OneProject.NeuroNet
{
    class Neuron
    {
        private NeuronType _type; // тип нейрона
        private double[] _weights; // массив весов
        private double[] _inputs; // его входы
        private double _output; // его выход
        private double _derivative; // производная функции активации
        // константы для функции активации
        private double a = 0.01; // для leakyRelu

        public double[] Weights { get => _weights; set => _weights = value; }

        public double[] Inputs
        {
            get { return _inputs; }
            set { _inputs = value; }
        }
        public double Output { get => _output; }
        public double Derivative { get => _derivative; }
        public Neuron(double[] weights, NeuronType type)
        {
            _type = type;
            _weights = weights;
        }
        public void Activator(double[] i, double[] w)
        {
            double sum = w[0];
            for(int m = 0; m < i.Length; m++)
            {
                sum += i[m] * w[m + 1];
            }
            /*switch (_type)
            {
                case NeuronType.Hidden:
                    _output = LeakyReLU(sum);
                    _derivative = LeakyReLU_Derivativator;
                    break;
                case NeuronType.Output:
                    _output = Exp(sum);
                    break;
                default:
                    break;
            }*/

            switch (_type)
            {
                case NeuronType.Hidden:
                    _output = LeakyReLU(sum);
                    break;
                case NeuronType.Output:
                    _output = Exp(sum);
                    break;
                default:
                    break;
            }

        }
        // Функция активации Leaky ReLU
        public double LeakyReLU(double x)
        {
            return x > 0 ? x : a * x;
        }
    }
}

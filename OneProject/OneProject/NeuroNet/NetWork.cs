using System;

namespace OneProject.NeuroNet
{
    class NetWork
    {
        private InputLayer input_layer = null;
        private HiddenLayer hidden_layer1 = new HiddenLayer(70, 15, NeuronType.Hidden, nameof(hidden_layer1));
        private HiddenLayer hidden_layer2 = new HiddenLayer(30, 70, NeuronType.Hidden, nameof(hidden_layer2));
        private OutputLayer output_layer = new OutputLayer(10, 30, NeuronType.Output, nameof(output_layer));

        public double[] fact = new double[10];

        private double[] e_error_avr;

        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }

        public NetWork() { }

        public void ForwardPass( NetWork net, double[] netInputs)
        {
            net.hidden_layer1.Data = netInputs;
            net.hidden_layer1.Recognize(null, net.hidden_layer2);
            net.hidden_layer2.Recognize(null, net.output_layer);
            net.output_layer.Recognize(net, null);
        }
        public void Train(NetWork net)
        {
            int epoches = 70; // кол-во эпох обучения
            net.input_layer = new InputLayer(NetworkMode.Train); //инициализация входного слоя
            double tmpSumError; // временная переменная суммы ошибок
            double[] errors; // массив входного сигнала ошибки выходного
            double[] temp_gsums1; // сумма локальных градиентов для 1 скрытого слоя
            double[] temp_gsums2; // сумма локальных градиентов для 2 скрытого слоя

            e_error_avr = new double[epoches];
            for(int k = 0; k < epoches; k++)
            {
                e_error_avr[k] = 0;
                net.input_layer.Shuffling_Array_Roms(net.input_layer.Trainset);
                for(int i = 0; i < net.input_layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15];
                    for(int j = 0;j < tmpTrain.Length; j++)
                    {
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1];
                    }
                    ForwardPass(net, tmpTrain);

                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for(int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Trainset[i, 0])
                            errors[x] = 1.0 - net.fact[x];
                        else
                            errors[x] = -net.fact[x];
                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length;

                    temp_gsums2 = net.output_layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] /= net.input_layer.Trainset.GetLength(0);
            }
            net.input_layer = null;

            net.hidden_layer1.WeightInitialize(MemoryMode.SET, nameof(hidden_layer1) + "_memory.csv");
        }

    }
}

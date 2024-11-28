using System;
using System.IO;


namespace OneProject.NeuroNet
{
    class InputLayer
    {
        private Random random = new Random();

        private double[,] trainset = new double[100, 16]; //массив обучающей выборки
        private double[,] testset = new double[10, 16]; // тестовой выборки

        public double[,] Trainset { get => trainset; }
        public double[,] Testset { get => testset; }

        public InputLayer(NetworkMode nm)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory;
            string[] tmpStr;
            string[] tmpArrStr;
            double[] tmpArr;
            switch (nm)
            {
                case NetworkMode.Train: // tak je dlya test
                    tmpArrStr = File.ReadAllLines(path + "train.txt");
                    for (int i = 0; i < tmpArrStr.Length; i++)
                    {
                        tmpStr = tmpArrStr[i].Split();
                        tmpArr = new double[tmpStr.Length];
                        for( int j = 0; j < tmpArrStr.Length; j++)
                        {
                            tmpArr[j] = double.Parse(tmpStr[j], System.Globalization.CultureInfo.InvariantCulture);

                        }
                    }
                    for (int n = trainset.GetLength(0) - 1; n >= 1; n--)
                    {
                        int j = random.Next(n + 1);
                        double[] temp = new double[trainset.GetLength(1)];
                        for(int i = 0; i < trainset.GetLength(1); i++)
                        {
                            temp[i] = trainset[n, i];
                        }
                        for (int i = 0; i < trainset.GetLength(1); i++)
                        {
                            trainset[n, i] = trainset[j, i];
                            trainset[j, i] = temp[i];
                        }
                    }
                    break;
                    //
                    //дописать для кнопки тест

            }
        }
        //перетасовка строк массива методом Фишера-Йетса
        public void Shuffling_Array_Roms(double[,] arr)
        {
            int j; // номер случайно выбранной строки
            Random random = new Random();
            double[] temp = new double[arr.GetLength(1)]; //вспомогательный массив
            for(int n = arr.GetLength(0) - 1; n >= 1; n--) // цикл перебора строк снизу вверх
            {
                j = random.Next(n + 1); //выбор случайной строки из выше расположенных строк
                for(int i = 0; i < arr.GetLength(1); i++) //цикл копирования н-ой строки
                {
                    temp[i] = arr[n, i];
                }
                for(int i = 0; i < arr.GetLength(1); i++) // перестановки двух строк
                {
                    arr[n, i] = arr[j, i]; 
                    arr[j, i] = temp[i];
                }
            }
        }
    }
}

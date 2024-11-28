
namespace OneProject.NeuroNet
{
    enum NeuronType
    {
        Hidden,//нейрон скрытого слоя
        Output //нейрон выходного слоя
    }
    enum NetworkMode
    {
        Train, 
        Test,
        Recogn
    }
    enum MemoryMode
    {
        GET,
        SET,
        INIT
    }
}

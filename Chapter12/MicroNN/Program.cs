using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MicroNN
{
    class Program
    {
        static void Main(string[] args)
        {
            var n1 = new Neuron(1);
            var n2 = new Neuron(1);
            var n3 = new Neuron(1);
            n1.Fires += n3.Signal;
            n2.Fires += n3.Signal;
            n1.Reset();
            n2.Reset();
            n3.Reset();

        }
    }

    internal sealed class Neuron
    {
        readonly double _threshold;
        private double _signalReceived;

        public Neuron(double threshold)
        {
            _threshold = threshold;
            Reset();
        }

        public void Reset() { _signalReceived = 0; }
        public event Action<double> Fires = delegate { };
        public void Signal(double strength)
        {
            if ((_signalReceived += strength) >= _threshold)
                Fires(.5);
        }
    }

   
}

#region License
/*
    Autoencoders library.
    Copyright (C) 2007 Thomas Benjamin Thompson

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/
#endregion
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace Autoencoders
{
    public class AutoencoderLearningRate
    {
        #region Vars
        internal List<double> prelrbiases = new List<double>();
        internal List<double> prelrweights = new List<double>();
        internal List<double> premombiases = new List<double>();
        internal List<double> premomweights = new List<double>();
        internal List<double> finelrbiases = new List<double>();
        internal List<double> finelrweights = new List<double>();
        #endregion

        #region Accessors
        public double GetPreTrainingBiasLR(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, prelrbiases.Count);
            return prelrbiases[PWhich];
        }
        public double GetPreTrainingWeightLR(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, prelrweights.Count);
            return prelrweights[PWhich];
        }
        public double GetPreTrainingBiasMom(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, premombiases.Count);
            return premombiases[PWhich];
        }
        public double GetPreTrainingWeightMom(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, premomweights.Count);
            return premomweights[PWhich];
        }
        public double GetFineTuningBiasLR(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, finelrbiases.Count);
            return finelrbiases[PWhich];
        }
        public double GetFineTuningWeightLR(int PWhich)
        {
            Utility.WithinBounds("Index out of bounds(Autoencoder)", PWhich, finelrweights.Count);
            return finelrweights[PWhich];
        }
        #endregion

        #region Save/Load
        internal void Save(TextWriter PFile)
        {
            Utility.SaveArray(prelrbiases.ToArray() , PFile);
            Utility.SaveArray(prelrweights.ToArray() , PFile);
            Utility.SaveArray(premombiases.ToArray() , PFile);
            Utility.SaveArray(premomweights.ToArray() , PFile);
            Utility.SaveArray(finelrbiases.ToArray() , PFile);
            Utility.SaveArray(finelrweights.ToArray() , PFile);
        }
        internal void Load(TextReader PFile)
        {
            prelrbiases.AddRange(Utility.LoadArray(PFile));
            prelrweights.AddRange(Utility.LoadArray(PFile));
            premombiases.AddRange(Utility.LoadArray(PFile));
            premomweights.AddRange(Utility.LoadArray(PFile));
            finelrbiases.AddRange(Utility.LoadArray(PFile));
            finelrweights.AddRange(Utility.LoadArray(PFile));
        }
        #endregion
    }
}

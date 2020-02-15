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
    public class AutoencoderWeights: ICloneable
    {
        private int numweightsets;
        private RBMWeightSet[] weights;
        public AutoencoderWeights()
        {

        }
        public AutoencoderWeights(int PNumLayers, RBMLayer[] PLayers, IWeightInitializer PWInitializer)
        {
            numweightsets = PNumLayers - 1;
            weights = new RBMWeightSet[numweightsets];
            for (int i = 0; i < numweightsets; i++)
            {
                weights[i] = new RBMWeightSet(PLayers[i].Count, PLayers[i + 1].Count, PWInitializer);
            }
        }
        public int NumWeightSets
        {
            get
            {
                return numweightsets;
            }
        }
        public RBMWeightSet GetWeightSet(int PPreSynapticLayer)
        {
            Utility.WithinBounds("Invalid weight set index!", PPreSynapticLayer, numweightsets);
            return weights[PPreSynapticLayer];
        }
        public RBMWeightSet GetRecogntionWeightSet(int PPreSynapticLayer)
        {
            Utility.WithinBounds("Invalid weight set index!", PPreSynapticLayer, numweightsets / 2);
            return weights[PPreSynapticLayer];
        }
        public RBMWeightSet GetGenerativeWeightSet(int PPreSynapticLayer)
        {
            Utility.WithinBounds("Invalid weight set index!", PPreSynapticLayer, numweightsets / 2, numweightsets);
            return weights[PPreSynapticLayer];
        }

        #region Save/Load
        internal void Save(TextWriter PFile)
        {
            PFile.WriteLine(numweightsets);
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                weights[i].Save(PFile);
            }
        }
        internal void Load(TextReader PFile)
        {
            numweightsets = int.Parse(PFile.ReadLine());
            weights = new RBMWeightSet[numweightsets];
            for(int i = 0;i < numweightsets;i++)
            {
                weights[i] = RBMWeightSet.Load(PFile);
            }
        }
        #endregion

        #region ICloneable Members

        public object Clone()
        {
            AutoencoderWeights retval = new AutoencoderWeights();
            retval.numweightsets = numweightsets;
            retval.weights = new RBMWeightSet[numweightsets];
            for (int i = 0; i < numweightsets; i++)
            {
                retval.weights[i] = (RBMWeightSet)weights[i].Clone();
            }
            return retval;
        }

        #endregion
    }
}

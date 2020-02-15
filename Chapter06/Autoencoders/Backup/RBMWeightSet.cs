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
    public class RBMWeightSet: ICloneable
    {
        #region Vars
        private int presize;
        private int postsize;
        private double[][] weights;
        private double[][] weightchanges;
        #endregion

        private RBMWeightSet()
        {

        }
        public RBMWeightSet(int PPreSynapticLayerSize, int PPostSynapticLayerSize, IWeightInitializer PWeightInit)
        {
            presize = PPreSynapticLayerSize;
            postsize = PPostSynapticLayerSize;
            weights = new double[presize][];
            weightchanges = new double[presize][];
            for (int i = 0; i < presize; i++)
            {
                weights[i] = new double[postsize];
                weightchanges[i] = new double[postsize];
                Utility.ZeroArray(weightchanges[i]);
                for (int j = 0; j < postsize; j++)
                {
                    weights[i][j] = PWeightInit.InitializeWeight();
                }
            }
        }

        #region WeightModification
        public void ModifyWeight(int PPre, int PPost, double PAmount)
        {
            CheckSynapseExists(PPre, PPost);
            weightchanges[PPre][PPost] = PAmount;
            weights[PPre][PPost] += PAmount;
        }
        public void SetWeight(int PPre, int PPost, double PValue)
        {
            CheckSynapseExists(PPre, PPost);
            weightchanges[PPre][PPost] = PValue - weights[PPre][PPost];
            weights[PPre][PPost] = PValue;
        }
        #endregion

        #region Accessors
        public double GetWeight(int PPre, int PPost)
        {
            CheckSynapseExists(PPre, PPost);
            return weights[PPre][PPost];
        }
        public double GetWeightChange(int PPre, int PPost)
        {
            CheckSynapseExists(PPre, PPost);
            return weightchanges[PPre][PPost];
        }
        public int PreSynapticLayerSize
        {
            get
            {
                return presize;
            }
        }
        public int PostSynapticLayerSize
        {
            get
            {
                return postsize;
            }
        }
        #endregion

        private void CheckSynapseExists(int PPre, int PPost)
        {
            Utility.WithinBounds("Pre-synaptic weight index out of bounds! K.O.", PPre, presize);
            Utility.WithinBounds("Post-synaptic weight index out of bounds! K.O.", PPost, postsize);
        }
        public object Clone()
        {
            RBMWeightSet newweights = new RBMWeightSet(presize, postsize, new ZeroWeightInitializer());
            for (int i = 0; i < presize; i++)
            {
                for (int j = 0; j < postsize; j++)
                {
                    newweights.SetWeight(i, j, weights[i][j]);
                }
            }
            return newweights;
        }

        #region Save/Load
        internal void Save(TextWriter PFile)
        {
            PFile.WriteLine(presize);
            for (int i = 0; i < presize; i++)
            {
                Utility.SaveArray(weights[i], PFile);
            }
        }
        internal static RBMWeightSet Load(TextReader PFile)
        {
            RBMWeightSet retval = new RBMWeightSet();
            retval.presize = int.Parse(PFile.ReadLine());
            retval.weights = new double[retval.presize][];
            retval.weightchanges = new double[retval.presize][];
            for (int i = 0; i < retval.presize; i++)
            {
                retval.weights[i] = Utility.LoadArray(PFile);
            }
            retval.postsize = retval.weights[0].GetLength(0);
            for (int i = 0; i < retval.presize; i++)
            {
                retval.weightchanges[i] = new double[retval.postsize];
            }
            return retval;
        }
        #endregion
    }
}

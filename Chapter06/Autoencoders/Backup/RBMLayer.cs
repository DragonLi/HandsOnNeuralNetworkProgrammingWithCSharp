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
    public abstract class RBMLayer: ICloneable
    {
        protected double[] state;
        protected double[] bias;
        protected double[] biaschange;
        protected double[] activity;
        protected int numneurons = 0;
        public RBMLayer()
        {

        }
        public RBMLayer(int PSize)
        {
            if (PSize <= 0)
            {
                throw new Exception("Can't have a layer with no neurons!");
            }
            numneurons = PSize;
            state = new double[numneurons];
            bias = new double[numneurons];
            biaschange = new double[numneurons];
            activity = new double[numneurons];
            for (int i = 0; i < PSize; i++)
            {
                state[i] = 0;
                bias[i] = 0;
                biaschange[i] = 0;
                activity[i] = 0;
            }
        }

        public void SetStateBypass(int PWhich, double PState)
        {
            WithinBounds(PWhich);
            state[PWhich] = PState;
            activity[PWhich] = PState;
        }
        public double GetState(int PWhich)
        {
            WithinBounds(PWhich);
            return state[PWhich];
        }
        public void SetBias(int PWhich, double PBias)
        {
            WithinBounds(PWhich);
            bias[PWhich] = PBias;
        }
        public double GetBias(int PWhich)
        {
            WithinBounds(PWhich);
            return bias[PWhich];
        }
        public void SetBiasChange(int PWhich, double PBiasChange)
        {
            WithinBounds(PWhich);
            biaschange[PWhich] = PBiasChange;
        }
        public double GetBiasChange(int PWhich)
        {
            WithinBounds(PWhich);
            return biaschange[PWhich];
        }
        public double GetActivity(int PWhich)
        {
            WithinBounds(PWhich);
            return activity[PWhich];
        }
        public double[] GetStates()
        {
            return (double[])state.Clone();
        }
        public double[] GetActivities()
        {
            return (double[])activity.Clone();
        }

        public int Count
        {
            get
            {
                return numneurons;
            }
        }
        public void WithinBounds(int PWhich)
        {
            if (PWhich < 0 || PWhich >= numneurons)
            {
                throw new Exception("Index out of bounds!!!!! GOOFBALL");
            }
        }
        public abstract void SetState(int PWhich, double PInput);

        #region ICloneable Members

        public abstract object Clone();

        #endregion

        #region Save/Load
        internal void Save(TextWriter PFile)
        {
            PFile.WriteLine(numneurons);
            Utility.SaveArray(bias, PFile);
        }
        internal void Load(TextReader PFile)
        {
            numneurons = int.Parse(PFile.ReadLine());
            bias = Utility.LoadArray(PFile);
            biaschange = new double[numneurons];
            activity = new double[numneurons];
            state = new double[numneurons];
            for (int i = 0; i < numneurons; i++)
            {
                biaschange[i] = 0;
                activity[i] = 0;
                state[i] = 0;
            }
        }
        #endregion
    }
}

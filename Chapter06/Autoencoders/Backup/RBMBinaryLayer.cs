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
namespace Autoencoders
{
    public class RBMBinaryLayer: RBMLayer
    {
        public RBMBinaryLayer()
        {

        }
        public RBMBinaryLayer(int PSize)
            :base(PSize)
        {
        }
        public override void SetState(int PWhich, double PInput)
        {
            WithinBounds(PWhich);
            activity[PWhich] = 1 / (1 + Math.Exp((-((PInput + bias[PWhich])))));
            if (Utility.NextDouble() < activity[PWhich])
            {
                state[PWhich] = 1;
            }
            else
            {
                state[PWhich] = 0;
            }
        }

        public override object Clone()
        {
            RBMBinaryLayer retval = new RBMBinaryLayer(numneurons);
            retval.state = (double[])state.Clone();
            retval.bias = (double[])bias.Clone();
            retval.biaschange = (double[])biaschange.Clone();
            retval.activity = (double[])activity.Clone();
            return retval;
        }
    }
}

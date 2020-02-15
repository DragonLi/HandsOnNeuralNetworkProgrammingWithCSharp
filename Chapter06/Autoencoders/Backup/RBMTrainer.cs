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
    public struct TrainingData
    {
        public double[] posvis;
        public double[] poshid;
        public double[] negvis;
        public double[] neghid;

        public void Zero()
        {
            Utility.ZeroArray(posvis);
            Utility.ZeroArray(poshid);
            Utility.ZeroArray(negvis);
            Utility.ZeroArray(neghid);
        }

        public void Scalar(double PScalar)
        {
            Utility.ScaleArray(posvis, PScalar);
            Utility.ScaleArray(poshid, PScalar);
            Utility.ScaleArray(negvis, PScalar);
            Utility.ScaleArray(neghid, PScalar);
        }
    }
    public static class RBMTrainer
    {
        private static RBMLearningRate learnrate;
        private static RBMWeightSet weightset;
        public static void Train(RBMLayer PLayerVis, RBMLayer PLayerHid, TrainingData PData
                                    , RBMLearningRate PLearnRate, RBMWeightSet PWeightSet)
        {
            weightset = PWeightSet;
            learnrate = PLearnRate;
            for (int i = 0; i < PLayerVis.Count; i++)
            {
                for (int j = 0; j < PLayerHid.Count; j++)
                {
                    TrainWeight(i, j, CalculateTrainAmount(PData.posvis[i], PData.poshid[j]
                        , PData.negvis[i], PData.neghid[j]));
                }
                TrainBias(PLayerVis, i, PData.posvis[i], PData.negvis[i]);
            }
            for (int j = 0; j < PLayerHid.Count; j++)
            {
                TrainBias(PLayerHid, j, PData.poshid[j], PData.neghid[j]);
            }
        }

        private static double CalculateTrainAmount(double PPosVis, double PPosHid, double PNegVis, double PNegHid)
        {
            return ((PPosVis * PPosHid) - (PNegVis * PNegHid));
        }

        private static void TrainWeight(int PWhichVis, int PWhichHid, double PTrainAmount)
        {
            weightset.ModifyWeight(PWhichVis, PWhichHid, 
                                 (learnrate.momweights * weightset.GetWeightChange(PWhichVis, PWhichHid))
                                  + (learnrate.lrweights * PTrainAmount)
                                  - (0.0002 * weightset.GetWeight(PWhichVis, PWhichHid)));
        }
        private static void TrainBias(RBMLayer PLayer, int PWhich, double PPosPhase, double PNegPhase)
        {
            double biaschange = (learnrate.mombiases * PLayer.GetBiasChange(PWhich)) + (learnrate.lrbiases * (PPosPhase - PNegPhase));
            PLayer.SetBiasChange(PWhich, biaschange);
            PLayer.SetBias(PWhich, PLayer.GetBias(PWhich) + biaschange);
        }
    }
}

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
using Autoencoders;

namespace Autoencoders
{
    public class AutoencoderBuilder
    {
        private List<RBMLayer> layers = new List<RBMLayer>();
        private AutoencoderLearningRate learnrate = new AutoencoderLearningRate();
        private IWeightInitializer weightinitializer = new GaussianWeightInitializer();
        public AutoencoderBuilder()
        {

        }

        public void AddBinaryLayer(int PSize)
        {
            AddLayer(new RBMBinaryLayer(PSize));
        }
        public void AddGaussianLayer(int PSize)
        {
            AddLayer(new RBMGaussianLayer(PSize));
        }
        public void SetPreTrainingLearningRateWeights(int PWhich, double PLR)
        {
            learnrate.prelrweights[PWhich] = PLR;
        }
        public void SetPreTrainingLearningRateBiases(int PWhich, double PLR)
        {
            learnrate.prelrbiases[PWhich] = PLR;
        }
        public void SetPreTrainingMomentumWeights(int PWhich, double PMom)
        {
            learnrate.premomweights[PWhich] = PMom;
        }
        public void SetPreTrainingMomentumBiases(int PWhich, double PMom)
        {
            learnrate.premombiases[PWhich] = PMom;
        }
        public void SetFineTuningLearningRateWeights(int PWhich, double PLR)
        {
            learnrate.finelrweights[PWhich] = PLR;
        }
        public void SetFineTuningLearningRateBiases(int PWhich, double PLR)
        {
            learnrate.finelrbiases[PWhich] = PLR;
        }

        private void AddLayer(RBMLayer PLayer)
        {
            learnrate.prelrbiases.Add(0.001);
            learnrate.premombiases.Add(0.5);
            learnrate.finelrbiases.Add(0.001);
            if(layers.Count >= 1)
            {
                learnrate.prelrweights.Add(0.001);
                learnrate.premomweights.Add(0.5);
                learnrate.finelrweights.Add(0.001);
            }
            layers.Add(PLayer);
        }

        public Autoencoder Build()
        {
            return new Autoencoder(layers, learnrate, weightinitializer);
        }
    }
}

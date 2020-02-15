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
    public class RBM
    {

        private RBMLayer visibles;
        private RBMLayer hiddens;
        private RBMWeightSet weights;
        private RBMLearningRate learnrate;
        private TrainingData trainingdata;
        private int numvisibles;
        private int numhiddens;

        public RBMWeightSet Weights
        {
            get { return weights; }
        }
        public double[] VisibleData
        {
            get
            {
                double[] retval = new double[numvisibles];
                for (int i = 0; i < numvisibles; i++)
                {
                    retval[i] = visibles.GetState(i);
                }
                return retval;
            }
        }
        public double[] VisibleActivity
        {
            get
            {
                double[] retval = new double[numvisibles];
                for (int i = 0; i < numvisibles; i++)
                {
                    retval[i] = visibles.GetActivity(i);
                }
                return retval;
            }
        }
        public double[] HiddenData
        {
            get
            {
                double[] retval = new double[numhiddens];
                for (int i = 0; i < numhiddens; i++)
                {
                    retval[i] = hiddens.GetState(i);
                }
                return retval;
            }
        }
        public double[] HiddenActivity
        {
            get
            {
                double[] retval = new double[numhiddens];
                for (int i = 0; i < numhiddens; i++)
                {
                    retval[i] = hiddens.GetActivity(i);
                }
                return retval;
            }
        }
        public int NumVisibles
        {
            get
            {
                return numvisibles;
            }
        }
        public int NumHiddens
        {
            get
            {
                return numhiddens;
            }
        }
        public RBMLearningRate LearnRate
        {
            get
            {
                return learnrate;
            }
            set
            {
                learnrate = value;
            }
        }

        public RBM(RBMLayer PVisibles, RBMLayer PHiddens, RBMLearningRate PLearnRate, IWeightInitializer PWeightInit)
        {
            numvisibles = PVisibles.Count;
            numhiddens = PHiddens.Count; 
            InitLayers(PVisibles, PHiddens);
            InitWeights(PWeightInit);
            InitTrainingData();
            learnrate = PLearnRate;
        }
        public RBM(RBM PA)
        {
            visibles = (RBMLayer)PA.visibles.Clone();
            hiddens = (RBMLayer)PA.hiddens.Clone();
            weights = (RBMWeightSet)PA.weights.Clone();
            learnrate = PA.learnrate;
            trainingdata = PA.trainingdata;
            numvisibles = PA.numvisibles;
            numhiddens = PA.numhiddens;
        }
        private void InitLayers(RBMLayer PVisibles, RBMLayer PHiddens)
        {
            if (PVisibles == null)
            {
                throw new Exception("You need a visible layer...");
            }
            if (PHiddens == null)
            {
                throw new Exception("You need a hidden layer...");
            }
            if (PVisibles.Count <= 0)
            {
                throw new Exception("You need at least one visible neuron...");
            }
            if (PHiddens.Count <= 0)
            {
                throw new Exception("You need at least one hidden neuron...");
            }
            hiddens = PHiddens;
            visibles = PVisibles;
        }
        private void InitWeights(IWeightInitializer PWeightInit)
        {
            weights = new RBMWeightSet(numvisibles, numhiddens, PWeightInit);
            for (int i = 0; i < numvisibles; i++)
            {
                for (int j = 0; j < numhiddens; j++)
                {
                    weights.SetWeight(i, j, Utility.NextGaussian(0,0.1));
                }
            }
        }
        private void InitTrainingData()
        {
            trainingdata = new TrainingData();
            trainingdata.posvis = new double[numvisibles];
            Utility.ZeroArray(trainingdata.posvis);
            trainingdata.poshid = new double[numhiddens];
            Utility.ZeroArray(trainingdata.poshid);
            trainingdata.negvis = new double[numvisibles];
            Utility.ZeroArray(trainingdata.negvis);
            trainingdata.neghid = new double[numhiddens];
            Utility.ZeroArray(trainingdata.neghid);
        }

        public double CalculateError()
        {
            double error = 0;
            for (int i = 0; i < numvisibles; i++)
            {
                double temp = trainingdata.posvis[i] - trainingdata.negvis[i];
                error += temp * temp;
            }
            error /= numvisibles;
            return error;
        }

        public void Compress(double[] PData)
        {
            SetVisibleData(PData);
            UpdateHiddens();
        }
        public void Reconstruct(double[] PData)
        {
            SetHiddenData(PData);
            Reconstruct();
        }
        public void Reconstruct()
        {
            UpdateVisibles();
        }

        public double Train(double[][] PBatchData)
        {
            if (PBatchData == null)
            {
                throw new Exception("Bad training data! DOOF!");
            } 
            
            double error = 0;
            trainingdata.Zero();
            for (int i = 0; i < PBatchData.GetLength(0); i++)
            {
                SaveTrainingData(PBatchData[i]);
                error += CalculateError();
            }
            error /= PBatchData.GetLength(0);
            trainingdata.Scalar(1 / PBatchData.GetLength(0));
            PerformTraining();
            return error;
        }
        public double Train(double[] PData)
        {
            double[][] batch = new double[1][];
            batch[0] = PData;
            return Train(batch);
        }

        private void PerformTraining()
        {
            RBMTrainer.Train(visibles, hiddens, trainingdata, learnrate, weights);
            
        }
        private void SaveTrainingData(double[] PData)
        {
            PositivePhase(PData);

            NegativePhase();
        }
        private void PositivePhase(double[] PData)
        {
            SetVisibleData(PData);
            UpdateHiddens();
            Utility.AddArrays(trainingdata.posvis, VisibleActivity);
            Utility.AddArrays(trainingdata.poshid, HiddenActivity);
        }
        private void NegativePhase()
        {
            UpdateVisibles();
            UpdateHiddens();
            Utility.AddArrays(trainingdata.negvis, VisibleActivity);
            Utility.AddArrays(trainingdata.neghid, HiddenActivity);
        }


        private void SetVisibleData(double[] PData)
        {
            if (PData.GetLength(0) != numvisibles)
            {
                throw new Exception("Too little or too much initial data");
            }
            for (int i = 0; i < numvisibles; i++)
            {
                visibles.SetStateBypass(i, PData[i]);
            }
        }
        private void SetHiddenData(double[] PData)
        {
            if (PData.GetLength(0) != numhiddens)
            {
                throw new Exception("Too little or too much initial data");
            }
            for (int i = 0; i < numhiddens; i++)
            {
                hiddens.SetStateBypass(i, PData[i]);
            }
        }
        private void UpdateHiddens()
        {
            double input = 0;
            double[] states = visibles.GetStates();
            for (int i = 0; i < numhiddens; i++)
            {
                for (int j = 0; j < numvisibles; j++)
                {
                    input += weights.GetWeight(j, i) * states[j];
                }
                hiddens.SetState(i, input);
                input = 0;
            }
        }
        private void UpdateVisibles()
        {
            double input = 0;
            double[] states = hiddens.GetStates();
            for (int i = 0; i < numvisibles; i++)
            {
                for (int j = 0; j < numhiddens; j++)
                {
                    input += weights.GetWeight(i, j) * states[j];
                }
                visibles.SetState(i,input);
                input = 0;
            }
        }
    }
}
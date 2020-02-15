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
using System.IO;

namespace Autoencoders
{
    public class Autoencoder
    {
        #region Vars
        private int numlayers;
        private bool pretraining = true;
        private RBMLayer[] layers;
        private AutoencoderLearningRate learnrate;
        private AutoencoderWeights recognitionweights;
        private AutoencoderWeights generativeweights;
        private TrainingData[] trainingdata;
        private List<IErrorObserver> errorobservers;
        #endregion

        #region Initialization
        private Autoencoder()
        {
        }
        internal Autoencoder(List<RBMLayer> PLayers, AutoencoderLearningRate PTrainingInfo
            , IWeightInitializer PWInitializer)
        {
            numlayers = PLayers.Count;
            layers = PLayers.ToArray();
            learnrate = PTrainingInfo;
            recognitionweights = new AutoencoderWeights(numlayers, layers, PWInitializer);
            generativeweights = new AutoencoderWeights(numlayers, layers, PWInitializer);
            errorobservers = new List<IErrorObserver>();
            InitBiases(PWInitializer);
            InitTrainingData();
        }

        private void InitBiases(IWeightInitializer PWInitializer)
        {
            for (int i = 0; i < numlayers; i++)
            {
                for (int j = 0; j < layers[i].Count; j++)
                {
                    layers[i].SetBias(j, PWInitializer.InitializeBias());
                }
            }
        }
        private void InitTrainingData()
        {
            trainingdata = new TrainingData[numlayers - 1];
            for (int i = 0; i < numlayers - 1; i++)
            {
                trainingdata[i].posvis = new double[layers[i].Count];
                Utility.ZeroArray(trainingdata[i].posvis);
                trainingdata[i].poshid = new double[layers[i + 1].Count];
                Utility.ZeroArray(trainingdata[i].poshid);
                trainingdata[i].negvis = new double[layers[i].Count];
                Utility.ZeroArray(trainingdata[i].negvis);
                trainingdata[i].neghid = new double[layers[i + 1].Count];
                Utility.ZeroArray(trainingdata[i].neghid);
            }
        }
        #endregion

        #region Accessors
        public int NumLayers
        {
            get
            {
                return numlayers;
            }
        }
        public AutoencoderWeights GetRecognitionWeights()
        {
            return recognitionweights;
        }
        public AutoencoderWeights GetGenerativeWeights()
        {
            return generativeweights;
        }
        public RBMLayer GetLayer(int PWhichLayer)
        {
            Utility.WithinBounds("Layer index out of bounds! DORK", PWhichLayer, numlayers);
            return layers[PWhichLayer];
        }
        public AutoencoderLearningRate LearningRate
        {
            get
            {
                return learnrate;
            }
        }
        #endregion

        #region PreTraining
        public void PreTrain(int PWhichLayer, double[] PData)
        {
            double[][] sentdata = new double[1][];
            sentdata[0] = PData;
            PreTrain(PWhichLayer, sentdata, 1);
        }
        public void PreTrain(int PWhichLayer, double[][] PData, int PBatchSize)
        {
            if (!pretraining)
            {
                throw new Exception("You have already called PreTrainingComplete()!"
                    + " You can't pretrain anymore");
            }
            Utility.WithinBounds("Layer to pretrain is invalid!", PWhichLayer, numlayers);
            Utility.WithinBounds("Invalid pre training batch size!", PBatchSize, PData.GetLength(0) + 1);
            double[][][] batches = CreateBatches(CalculateToLayer(0,PWhichLayer, PData), PBatchSize);
            for (int i = 0; i < batches.GetLength(0); i++)
            {
                PreTrainCollectData(PWhichLayer, batches[i]);
                PerformPreTraining(PWhichLayer);
                CalculateError(trainingdata[PWhichLayer].posvis, trainingdata[PWhichLayer].negvis);
            }
        }
        public void PreTrainingComplete()
        {
            if (!pretraining)
            {
                throw new Exception("You have already called PreTrainingComplete()!"
                    + " Don't call it twice!");
            }
            pretraining = false;
            for (int i = 0; i < numlayers; i++)
            {
                generativeweights = (AutoencoderWeights)recognitionweights.Clone();
            }
        }

        private void PreTrainCollectData(int PWhichLayer, double[][] PData)
        {
            trainingdata[PWhichLayer].Zero();
            for(int i = 0;i < PData.GetLength(0);i++)
            {
                SetLayerData(PWhichLayer,PData[i]);
                UpdateLayer(PWhichLayer + 1, true, recognitionweights);
                Utility.AddArrays(trainingdata[PWhichLayer].posvis, layers[PWhichLayer].GetActivities());
                Utility.AddArrays(trainingdata[PWhichLayer].poshid, layers[PWhichLayer + 1].GetActivities());
                UpdateLayer(PWhichLayer, false, recognitionweights);
                UpdateLayer(PWhichLayer + 1, true, recognitionweights);
                Utility.AddArrays(trainingdata[PWhichLayer].negvis, layers[PWhichLayer].GetActivities());
                Utility.AddArrays(trainingdata[PWhichLayer].neghid, layers[PWhichLayer + 1].GetActivities());
            }
            trainingdata[PWhichLayer].Scalar(1 / PData.GetLength(0));
        }
        private void PerformPreTraining(int PPreSynapticLayer)
        {
            RBMLearningRate sentlearnrate = new RBMLearningRate(learnrate.prelrweights[PPreSynapticLayer]
                                                            , learnrate.prelrbiases[PPreSynapticLayer]
                                                            , learnrate.premomweights[PPreSynapticLayer]
                                                            , learnrate.premombiases[PPreSynapticLayer]);
            RBMTrainer.Train(layers[PPreSynapticLayer], layers[PPreSynapticLayer + 1], trainingdata[PPreSynapticLayer]
                , sentlearnrate, recognitionweights.GetWeightSet(PPreSynapticLayer));
        }

        private void UpdateLayerBackwardPreTrain(int PWhich)
        {
            Utility.WithinBounds("Cannot update this layer!!!", PWhich, numlayers - 1);
            RBMLayer thislayer = layers[PWhich];
            RBMLayer nextlayer = layers[PWhich + 1];
            double input = 0;
            double[] states = nextlayer.GetStates();
            for (int i = 0; i < thislayer.Count; i++)
            {
                for (int j = 0; j < nextlayer.Count; j++)
                {
                    input += recognitionweights.GetWeightSet(PWhich).GetWeight(i, j) * states[j];
                }
                thislayer.SetState(i, input);
                input = 0;
            }
        }
        #endregion

        #region FineTuning
        /*
         * Implements Wake-Sleep Training for an autoencoder. 
         */
        public void FineTune(double[] PData)
        {
            double[][] sentdata = new double[1][];
            sentdata[0] = PData;
            FineTune(sentdata, 1);
        }
        public void FineTune(double[][] PData, int PBatchSize)
        {
            if (pretraining)
            {
                throw new Exception("You haven't called PreTrainingComplete()!"
                    + " You can't fine tune yet!");
            }
            if (PBatchSize < 1 || PBatchSize > PData.GetLength(0))
            {
                throw new Exception("Invalid pre training batch size!");
            }
            double[][][] batches = CreateBatches(PData, PBatchSize);
            for (int i = 0; i < batches.GetLength(0); i++)
            {
                FineTuneBatch(batches[i]);
            }
        }

        private void FineTuneBatch(double[][] PData)
        {
            for(int i = 0;i < PData.GetLength(0);i++)
            {
                Compress(PData[i]);
                WakePhase();
                Reconstruct();
                SleepPhase();
                CalculateError(PData[i], layers[0].GetActivities());
            }
        }

        private void WakePhase()
        {
            for (int i = 0; i < numlayers - 1; i++)
            {
                double[] visstates = layers[i].GetStates();
                double[] visact = layers[i].GetActivities();
                double[] hidstates = layers[i + 1].GetStates();
                double[] hidact = layers[i + 1].GetActivities();
                double curlearnrate = learnrate.finelrweights[i];
                for (int j = 0; j < layers[i].Count; j++)
                {
                    for (int k = 0; k < layers[i + 1].Count; k++)
                    {
                        generativeweights.GetWeightSet(i).ModifyWeight(j, k, curlearnrate *
                               CalculateFineTuneTrain(hidstates[k], visstates[j], visact[j]));
                    }
                }
            }
        }
        private void SleepPhase()
        {
            for (int i = 0; i < numlayers - 1; i++)
            {
                double[] visstates = layers[i].GetStates();
                double[] visact = layers[i].GetActivities();
                double[] hidstates = layers[i + 1].GetStates();
                double[] hidact = layers[i + 1].GetActivities();
                double curlearnrate = learnrate.finelrweights[i];
                for (int j = 0; j < layers[i].Count; j++)
                {
                    for (int k = 0; k < layers[i + 1].Count; k++)
                    {
                        recognitionweights.GetWeightSet(i).ModifyWeight(j, k, curlearnrate *
                               CalculateFineTuneTrain(visstates[j], hidstates[k], hidact[k]));
                    }
                }
            }
        }
        private double CalculateFineTuneTrain(double PPreState, double PPostState, double PPostActivity)
        {
            return PPreState * (PPostState - PPostActivity);
        }
        #endregion

        #region Running
        public void Compress(double[] PData)
        {
            if (PData == null)
            {
                throw new Exception("No null data allowed! COME ON DUDE!");
            }
            CalculateToLayer(0, numlayers - 1, PData);
        }
        public void Reconstruct(double[] PData)
        {
            if(PData == null)
            {
                throw new Exception("No null data allowed! COME ON DUDE!");
            }
            CalculateToLayer(numlayers - 1, 0, PData);
        }
        public void Reconstruct()
        {
            Reconstruct(layers[numlayers - 1].GetStates());
        }
        #endregion

        #region LayerCalculation
        private double[][] CalculateToLayer(int PStartLayer, int PEndLayer, double[][] PData)
        {
            int batchsize = PData.GetLength(0);
            double[][] newdata = new double[batchsize][];
            for (int i = 0; i < batchsize; i++)
            {
                newdata[i] = CalculateToLayer(PStartLayer, PEndLayer, PData[i]);
            }
            return newdata;
        }
        private double[] CalculateToLayer(int PStartLayer, int PEndLayer, double[] PData)
        {
            SetLayerData(PStartLayer, PData);
            return CalculateToLayer(PStartLayer, PEndLayer);
        }
        private double[] CalculateToLayer(int PStartLayer, int PEndLayer)
        {
            int looplimit = PEndLayer - PStartLayer;
            if (PStartLayer > PEndLayer)
            {
                looplimit = PStartLayer - PEndLayer;
            }
            for (int i = 1; i <= looplimit; i++)
            {
                if (PStartLayer < PEndLayer)
                {
                    UpdateLayer(PStartLayer + i, true, recognitionweights);
                }
                else
                {
                    UpdateLayer(PStartLayer - i, false, generativeweights);
                }
            }
            return layers[PEndLayer].GetActivities();
        }

        private void UpdateLayer(int PWhichLayer, bool PForward, AutoencoderWeights PWeights)
        {
            int beginlayer = PWhichLayer - 1;
            if (PForward)
            {
                Utility.WithinBounds("Cannot update this layer!!!", PWhichLayer, 1, numlayers);
            }
            else
            {
                Utility.WithinBounds("Cannot update this layer!!!", PWhichLayer, 0, numlayers - 1);
                beginlayer = PWhichLayer + 1;
            }
            RBMLayer thislayer = layers[PWhichLayer];
            RBMLayer previouslayer = layers[beginlayer];
            double input = 0;
            double[] states = previouslayer.GetStates();
            for (int i = 0; i < thislayer.Count; i++)
            {
                for (int j = 0; j < previouslayer.Count; j++)
                {
                    if (!PForward)
                    {
                        input += PWeights.GetWeightSet(beginlayer - 1).GetWeight(i, j) * states[j];
                    }
                    else
                    {
                        input += PWeights.GetWeightSet(beginlayer).GetWeight(j, i) * states[j];
                    }
                }
                thislayer.SetState(i, input);
                input = 0;
            }
        }

        private void SetLayerData(int PWhich, double[] PData)
        {
            Utility.WithinBounds("Layer index out of bounds! DORKOSAUR", PWhich, numlayers);
            if (PData.GetLength(0) != layers[PWhich].Count)
            {
                throw new Exception("Too little or too much initial data");
            }
            for (int i = 0; i < layers[PWhich].Count; i++)
            {
                layers[PWhich].SetStateBypass(i, PData[i]);
            }
        }
        private double[][][] CreateBatches(double[][] PData, int PBatchSize)
        {
            int numbatches = (int)Math.Ceiling(((double)PData.GetLength(0)) / ((double)PBatchSize));
            double[][][] batches = new double[numbatches][][];
            for(int i = 0;i < numbatches;i++)
            {
                batches[i] = new double[PBatchSize][];
                for(int j = 0;j < PBatchSize;j++)
                {
                    batches[i][j] = PData[(i * PBatchSize) + j];
                }
            }
            return batches;
        }
        #endregion

        #region ErrorObservation
        public List<IErrorObserver> ErrorObservers
        {
            get
            {
                return errorobservers;
            }
            set
            {
                if (value != null)
                {
                    errorobservers = value;
                }
            }
        }
        private void CalculateError(double[] POriginal, double[] PReconstruction)
        {
            int originallength = POriginal.GetLength(0);
            if (originallength != PReconstruction.GetLength(0))
            {
                throw new Exception("Tried to calculate error for different size lists.");
            }
            double error = 0;
            for (int i = 0; i < originallength; i++)
            {
                double temp = POriginal[i] - PReconstruction[i];
                error += temp * temp;
            }
            error /= originallength;
            SendError(error);
        }
        private void SendError(double PError)
        {
            for (int i = 0; i < errorobservers.Count; i++)
            {
                errorobservers[i].OnErrorCalculated(PError);
            }
        }
        #endregion

        #region Save/Load
        public void Save(string PFilename)
        {
            TextWriter file = new StreamWriter(PFilename);
            learnrate.Save(file);
            recognitionweights.Save(file);
            generativeweights.Save(file);
            file.WriteLine(numlayers);
            for (int i = 0; i < numlayers; i++)
            {
                if(layers[i].GetType() == typeof(RBMGaussianLayer))
                {
                    file.WriteLine("RBMGaussianLayer");
                }
                else if (layers[i].GetType() == typeof(RBMBinaryLayer))
                {
                    file.WriteLine("RBMBinaryLayer");
                }
                layers[i].Save(file);
            }
            file.WriteLine(pretraining);
            file.Close();
        }
        public static Autoencoder Load(string PFilename)
        {
            TextReader file = new StreamReader(PFilename);
            Autoencoder retval = new Autoencoder();
            retval.learnrate = new AutoencoderLearningRate();
            retval.learnrate.Load(file);
            retval.recognitionweights = new AutoencoderWeights();
            retval.recognitionweights.Load(file);
            retval.generativeweights = new AutoencoderWeights();
            retval.generativeweights.Load(file);
            retval.numlayers = int.Parse(file.ReadLine());
            retval.layers = new RBMLayer[retval.numlayers];
            for (int i = 0; i < retval.numlayers; i++)
            {
                string type = file.ReadLine();
                if (type == "RBMGaussianLayer")
                {
                    retval.layers[i] = new RBMGaussianLayer();
                }
                else if (type == "RBMBinaryLayer")
                {
                    retval.layers[i] = new RBMBinaryLayer();
                }
                retval.layers[i].Load(file);
            }
            retval.pretraining = bool.Parse(file.ReadLine());
            retval.InitTrainingData();
            retval.errorobservers = new List<IErrorObserver>();
            file.Close();
            return retval;
        }
        #endregion
    }
}

﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SwarmOps.Problems
{
    using System.Diagnostics;

    public class Beales : BenchmarkProblem
    {
        #region Constructors.

        /// <summary>
        /// Construct the object.
        /// </summary>
        /// <param name="dimensionality">Dimensionality of the problem (e.g. 20)</param>
        /// <param name="maxIterations">Max optimization iterations to perform.</param>
        public Beales(int dimensionality, int maxIterations)
            : base(dimensionality, -10, 10, 0, 0, 100)
        {
        }

        #endregion

        #region Base-class overrides.

        /// <summary>
        /// Name of the optimization problem.
        /// </summary>
        public override string Name => "Beales";

        /// <summary>
        /// Minimum possible fitness.
        /// </summary>
        public override double MinFitness => -5;

        /// <summary>
        /// Has the gradient has been implemented?
        /// </summary>
        public override bool HasGradient => false;

        /// <summary>
        /// Compute and return fitness for the given parameters.
        /// </summary>
        /// <param name="x">Candidate solution.</param>
        public override double Fitness(double[] x)
        {
            Debug.Assert(x != null && x.Length == Dimensionality);
            double f = (1.5 - x[0] + x[0] * x[1]) * (1.5 - x[0] + x[0] * x[1])
              + (2.25 - x[0] + x[0] * x[1] * x[1]) * (2.25 - x[0] + x[0] * x[1] * x[1])
              + (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]) * (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]);
            return f;
        }

        #endregion
    }
}

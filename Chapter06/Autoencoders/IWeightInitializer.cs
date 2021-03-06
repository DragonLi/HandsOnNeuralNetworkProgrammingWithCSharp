
namespace Autoencoders
{
    /// <summary>   Interface for weight initializer. </summary>
    public interface IWeightInitializer
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes the weight. </summary>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        double InitializeWeight();

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes the bias. </summary>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        double InitializeBias();
    }
}

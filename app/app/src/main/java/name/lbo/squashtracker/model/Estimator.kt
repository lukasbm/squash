package name.lbo.squashtracker.model

import org.opencv.core.Rect

interface Estimator {
    fun correct(position: Rect)
    fun predict(): Rect

}

object DoubleExponentialEstimator : Estimator {
    private const val DATA_SMOOTHING = 0.9f
    private const val TREND_SMOOTHING = 0.25f

    private var lastPosition = Rect()
    p


    fun correct(position: Rect) {
        lastPosition = position
    }

    // predicts a rectangle
    fun predict(): Rect {
        val smoothedX = smoothedValue(
            lastPosition.x.toFloat(),
            lastPosition.x.toFloat(),
            lastTrend
        )

        return Rect(
            predictionX - 1, predictionY - 1, lastPosition.width + 2, lastPosition.height + 2
        )
    }

    /**
     * Calculate the 'smoothed value' part of a double-exponential smoothing process.
     *
     * This is the 'level' part of the double-exponential smoothing process.
     *
     * @param observedTrue: Top-left corner of bounding rectangle which is assumed to be a true positive.
     * @param prevSmoothed: Smoothed top-left corner of a bounding rectangle from previous time-step.
     * @param prevTrend: Trend value from previous time-step.
     * @return: Smoothed estimate of top-left corner of bounding rectangle.
     */
    private fun smoothedValue(observedTrue: Float, prevSmoothed: Float, prevTrend: Float): Float {
        return DATA_SMOOTHING * observedTrue + (1 - DATA_SMOOTHING) * (prevSmoothed + prevTrend)
    }

    /**
     * Calculate the 'smoothed value' part of a double-exponential smoothing process.

    :param observedTrue: Observed true value at current time-step.
    :param prevSmoothed: Smoothed value at previous time-step.
    :param prevTrend: Previous trend value.
    :return: New smoothed value.
     */
    private fun calculateTrendEstimate(
        currSmoothed: Float,
        prevSmoothed: Float,
        prevTrend: Float
    ): Float {
        return TREND_SMOOTHING * (currSmoothed - prevSmoothed) + (1 - TREND_SMOOTHING) * prevTrend
    }
}

package name.lbo.squashtracker.model

import org.opencv.core.Point
import org.opencv.core.Rect

interface Estimator {
    fun correct(position: Rect)
    fun predict(): Rect
}

class DoubleExponentialEstimator(initialPos: Rect = Rect(), nextPos: Rect = Rect()) : Estimator {
    companion object {
        private const val DATA_SMOOTHING = 0.9
        private const val TREND_SMOOTHING = 0.25
    }

    // FIXME: clean this up without using circular buffer!
    private val positionBuffer = CircularBuffer<Rect>(2)
    private var previousSmoothed: Point
    private var previousTrend: Point

    init {
        positionBuffer.add(initialPos)
        positionBuffer.add(nextPos)
        previousSmoothed = positionBuffer[0].tl()
        previousTrend = positionBuffer[1].tl().subtract(positionBuffer[0].tl())
    }

    override fun correct(position: Rect) {
        positionBuffer.add(position)
    }

    // predicts a rectangle
    override fun predict(): Rect {
        val prev = positionBuffer.getLast()

        val smoothedX = smoothedValue(
            prev.x.toDouble(), previousSmoothed.x, previousTrend.x
        )
        val smoothedY = smoothedValue(
            prev.y.toDouble(), previousSmoothed.y, previousTrend.y
        )

        val trendX = calculateTrendEstimate(
            smoothedX, previousSmoothed.x, previousTrend.x
        )
        val trendY = calculateTrendEstimate(
            smoothedY, previousSmoothed.y, previousTrend.y
        )

        val predictionX = smoothedX + trendX
        val predictionY = smoothedY + trendY

        // Update the smoothed and trend values for the next prediction
        previousSmoothed = Point(predictionX, predictionY)
        previousTrend = Point(trendX, trendY)

        return Rect(
            predictionX.toInt() - 1, predictionY.toInt() - 1, prev.width + 2, prev.height + 2
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
    private fun smoothedValue(
        observedTrue: Double, prevSmoothed: Double, prevTrend: Double
    ): Double {
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
        currSmoothed: Double, prevSmoothed: Double, prevTrend: Double
    ): Double {
        return TREND_SMOOTHING * (currSmoothed - prevSmoothed) + (1 - TREND_SMOOTHING) * prevTrend
    }
}

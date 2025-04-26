package name.lbo.squashtracker

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

import kotlin.math.pow
import kotlin.math.sqrt

enum class ImageOperation(val displayName: String) {
    GAUSSIAN_BLUR("Gaussian Blur") {
        override fun apply(src: Mat, dst: Mat) {
            Imgproc.GaussianBlur(src, dst, Size(7.0, 7.0), 0.0)
        }
    },
    SOBEL("Sobel") {
        override fun apply(src: Mat, dst: Mat) {
            Imgproc.Sobel(src, dst, CvType.CV_16S, 1, 0, 3
                , 1.0, 0.0, Core.BORDER_REPLICATE)
        }
    },
    RESIZE("Resize") {
        override fun apply(src: Mat, dst: Mat) {
            Imgproc.resize(
                src,
                dst,
                Size(src.cols() / 2.0, src.rows() / 2.0)
            )
        }
    },
    ROTATE_90("Rotate 90°") {
        override fun apply(src: Mat, dst: Mat) {
            Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE)
        }
    };

    abstract fun apply(src: Mat, dst: Mat)

    companion object {
        fun fromDisplayName(name: String): ImageOperation? =
            entries.find { it.displayName == name }
    }
}

class ImageProcessor {
    fun applyOperation(src: Mat, dst: Mat, operation: ImageOperation) {
        operation.apply(src, dst)
    }
}


data class PerformanceMetrics(private val durationsNano: List<Long>) {
    private val toMillis = 1_000_000.0

    val average: Double
        get() = durationsNano.average() / toMillis

    val min: Double
        get() = durationsNano.min() / toMillis

    val max: Double
        get() = durationsNano.max() / toMillis

    val standardDeviation: Double
        get() {
            val mean = durationsNano.average()
            return sqrt(durationsNano.map { (it - mean).pow(2) }.average()) / toMillis
        }

    override fun toString(): String = buildString {
        append("Average: %.2f ms\n".format(average))
        append("Min: %.2f ms\n".format(min))
        append("Max: %.2f ms\n".format(max))
        append("Std Dev: %.2f ms".format(standardDeviation))
    }
}
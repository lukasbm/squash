package name.lbo.squashtracker

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc

/*
Preprocesses video frames.
Does foreground extraction, image filtering, etc.

To achieve this, the following procedure developed by

Saumil Sachdeva (http://resolver.tudelft.nl/uuid:758d345d-ecdf-478e-a534-a23300dbe877) is used:

• Three consecutive frames of the video are gathered
• All three frames are converted to grayscale images
• Noise reduction is performed using a gaussian filter
• Consecutive frames are combined via frame differencing
• The images are combined yet again with a boolean “and” to achieve a single image
• The image is thresholded to obtain a binary image
• Morphological operations are used to to enhance and bolster the moving components of the image.

The result of this procedure will be a binary image with the foreground (moving objects) extracted
from the background. The foreground consists of exactly our objects of interest - the moving players and the ball.

Reference: https://github.com/veedlaw/squash-drive-analyst/blob/master/detector.py
*/
object Detector {
    private val buffer = CircularBuffer<Mat>(3);
    private val frameDifferences = CircularBuffer<Mat>(2);
    private val dilationKernel = Mat(3, 3, CvType.CV_8U)

    fun isReady(): Boolean {
        return buffer.isFull()
    }

    // TODO: make sure src and dst are 8bit (u8) images
    fun process(src: Mat, dst: Mat) {
        /*
        Uses a sliding window approach in the frame buffer for differentiating moving parts of the image from static
        parts.

        Frames are received via the 'frame' parameter and after cleaning operations are added to the buffer.

        :param frame: A video frame
        :return: A binary image that has differentiated moving parts of the image from static parts.
         */
        buffer.add(src.clone())
        if (!buffer.isFull()) {
            return
        }

        // form new diff
        val newestDiff = Mat()  // TODO: make this a instance var, to save memory! otherwise it will be re-allocated every frame!
        Core.absdiff(buffer[0], buffer[1], newestDiff)
        frameDifferences.add(newestDiff)

        // combine it with the previous diff
        val combined = Mat()
        Core.bitwise_and(frameDifferences[0], frameDifferences[1], combined)

        // threshold
        val thresholded = Mat()
        val ret = Imgproc.threshold(
            combined,
            thresholded,
            0.0,
            255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )
        // If Otsu's thresholding picks a low threshold due to low amount of foreground pixels
        if (ret <= 8.0) {
            Imgproc.threshold(
                combined,
                thresholded,
                24.0,
                255.0,
                Imgproc.THRESH_BINARY
            )
        }

        // morphological operations
        morphologicalOperation(thresholded, dst, 9)
    }

    private fun morphologicalOperation(
        src: Mat,
        dst: Mat,
        iterations: Int,
    ) {
        // dilation followed by erosion
        val dilated = Mat()
        Imgproc.dilate(src, dilated, dilationKernel, Point(-1.0, -1.0), iterations)
        Imgproc.erode(dilated, dst, dilationKernel)
    }
}

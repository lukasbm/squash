package name.lbo.squashtracker

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc

class Tracker() {
    private val candidateHistory = CircularBuffer<Collection<Rect>>(7)

    private val avgArea = 25 * 24

    private var prevBestDist = 0
    private var distJumpCutoff = 100

    init {
        val dummyCandidate = Rect()
        // simulate first two frames having a candidate
        candidateHistory.add(setOf(dummyCandidate))
        candidateHistory.add(setOf(dummyCandidate))
    }

    // Finds the shortest path through sequences of ball candidates.
    private fun shortestPathCandidate()

    private fun joinNearbyBoundingBoxes(boxes: Collection<Rect>, joinDistanceX : Int = 5, joinDistanceY: Int = 10) : Collection<Rect> {
        val processed = arrayOfNulls<Boolean>(boxes.size)
        val newBoxes = mutableListOf<Rect>()

        boxes.forEachIndexed { i, r -> {

        } }

        return newBoxes
    }

    private fun getContours(frame: Mat, prediction: Rect): Collection<Rect> {
        val edges = Mat()
        Imgproc.Canny(frame, edges, 100.0, 200.0)
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(edges, contours, null, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE) // FIXME: or empty mat for hierarchy?
        return contours.map { Imgproc.boundingRect(it) }
    }
}
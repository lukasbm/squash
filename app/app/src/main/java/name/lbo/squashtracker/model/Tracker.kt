package name.lbo.squashtracker.model

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

object Tracker {
    private val candidateHistory = CircularBuffer<Collection<Rect>>(7)
    private val pairHistory = CircularBuffer<Collection<Pair<Rect, Rect>>>(6)
    private const val AVG_AREA = 25 * 24
    private val pathGraph = mutableMapOf<Rect, PathElement>() // this is across all frames

    data class PathElement(
        val from: Rect,
        val to: Rect,
        val distance: Double,
        val hops: Int,
    )

    init {
        val dummyCandidate = Rect()
        // simulate first two frames having a candidate
        candidateHistory.add(setOf(dummyCandidate))
        candidateHistory.add(setOf(dummyCandidate))
        pairHistory.add(findPairs(candidateHistory[0], candidateHistory[1]))
    }

    public fun selectMostPromisingCandidate(frame: Mat, prediction: Rect): Rect {
        // all new candidates for this frame
        val newCandidates = findBallCandidates(frame).ifEmpty { mutableListOf(prediction) }
        val lastCandidates = candidateHistory.getLast()
        candidateHistory.add(newCandidates)
        val pairs = findPairs(lastCandidates, newCandidates)
        pairHistory.add(pairs)
        // find most probable by path
        return findShortestPathCandidate(pairHistory.toList())
    }

    // finds the most probable candidate by looking at the path of the candidates
    // @param pairs: For each pair of subsequent frames a bag of pairs (connections)
    private fun findShortestPathCandidate(pairs: Collection<Collection<Pair<Rect, Rect>>>): Rect {
        return Rect() // TODO
    }

    // @param pairs: (from, to) highlighting connections of rects between subsequent frames.
    private fun addParisToGraph(pairs: Collection<Pair<Rect, Rect>>) {
        pairs.forEach { (from, to) ->
            val dist = from.tl().distanceTo(to.tl())
//            if pathGraph.containsKey(from) {
//                pathGraph.computeIfPresent(to) { key, value ->
//                    PathElement(
//                        from,
//                        to,
//                        value.distance + dist,
//                        value.hops + 1
//                    )
//                }
//            } else {
//
//            }
        }
    }

    private fun removeFromGraph(rects: Collection<Rect>) {

    }

    private fun findBallCandidates(frame: Mat): Collection<Rect> {
        val contours = getContours(frame)
        val ballCandidates =
            contours.filter { AVG_AREA * 0.3 < it.area() && it.area() < AVG_AREA * 3 }
        return ballCandidates.sortedBy { it.area() }.dropLast(2) // drop players
    }

    // finds pairs in the candidates of two subsequent frames
    // a good pair is one that has a similar area and is close to each other
    // finds a pair for each candidate in the `next` frame's rects
    private fun findPairs(
        prev: Collection<Rect>,
        next: Collection<Rect>
    ): Collection<Pair<Rect, Rect>> {
        val pairs = mutableListOf<Pair<Rect, Rect>>()

        for (n in next) {
            // FIXME: turn this into a shape penalty, where squareness of n and p are compared.
            val squareness = abs(n.width - n.height)
            // will always find something, even if its only the prediction.
            val bestPrev = prev.minBy { p ->
                val dist = sqrt(
                    (p.x - n.x).toDouble().pow(2) +
                            (p.y - n.y).toDouble().pow(2)
                )
                val distancePenalty = if (dist < 50) 0 else 50
                val sizePenalty = if (abs(p.area() - n.area()) < AVG_AREA * 0.5) 0 else 50
                dist + squareness + distancePenalty + sizePenalty
            }
            pairs.add(Pair(bestPrev, n))
        }

        return pairs
    }

    private fun joinNearbyBoundingBoxes(
        boxes: Collection<Rect>,
        joinDistanceX: Int = 5,
        joinDistanceY: Int = 10
    ): Collection<Rect> {
        val processed = arrayOfNulls<Boolean>(boxes.size)
        val newBoxes = mutableListOf<Rect>()

        // TODO: for rewrite: maybe use rect.tl and rect.bl and rect.contains
        for (i in boxes.indices) {
            if (processed[i] == true) continue
            val box = boxes.elementAt(i)
            var newBox = box
            for (j in i + 1 until boxes.size) {
                val otherBox = boxes.elementAt(j)
                if (processed[j] == true) continue
                if (abs(box.x - otherBox.x) < joinDistanceX && abs(box.y - otherBox.y) < joinDistanceY) {
                    newBox = Rect(
                        Point(
                            box.tl().x.coerceAtMost(otherBox.tl().x),
                            box.tl().y.coerceAtMost(otherBox.tl().y)
                        ),
                        Point(
                            box.br().x.coerceAtLeast(otherBox.br().x),
                            box.br().y.coerceAtLeast(otherBox.br().y)
                        )
                    )
                    processed[j] = true
                }
            }
            newBoxes.add(newBox)
        }

        return newBoxes
    }

    private fun getContours(frame: Mat): Collection<Rect> {
        val edges = Mat()
        Imgproc.Canny(frame, edges, 100.0, 200.0)
        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(
            edges,
            contours,
            null,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        ) // FIXME: or empty mat for hierarchy?
        val boundingBoxes = contours.map { Imgproc.boundingRect(it) }
        return joinNearbyBoundingBoxes(boundingBoxes)
    }
}
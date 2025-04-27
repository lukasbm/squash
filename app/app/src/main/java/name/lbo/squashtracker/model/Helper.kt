package name.lbo.squashtracker.model

import org.opencv.core.Point
import kotlin.math.pow
import kotlin.math.sqrt

// TODO: to squeeze out even more performance, Mat should be passed by reference, and reused.

class CircularBuffer<T>(private val maxSize: Int) {
    private val buffer: Array<Any?> = arrayOfNulls(maxSize)
    private var start = 0
    private var size = 0

//    fun add(item: T) {
//        val end = (start + size) % maxSize
//
//        if (buffer[end] is Mat && item is Mat) {
//            // Reuse the existing Mat by copying the new data into it
//            (buffer[end] as Mat).release()
//            item.copyTo(buffer[end] as Mat)
//        } else {
//            buffer[end] = item
//        }
//
//        if (size < maxSize) {
//            size++
//        } else {
//            start = (start + 1) % maxSize // Overwrite oldest
//        }
//    }

    fun add(item: T) {
        val end = (start + size) % maxSize
        buffer[end] = item

        if (size < maxSize) {
            size++
        } else {
            start = (start + 1) % maxSize // Overwrite oldest
        }
    }

    operator fun get(index: Int): T {
        require(index in 0 until size) { "Index out of bounds: $index" }
        @Suppress("UNCHECKED_CAST")
        return buffer[(start + index) % maxSize] as T
    }

    fun toList(): List<T> {
        return List(size) { get(it) }
    }

    fun getSize(): Int = size

    fun isFull(): Boolean = size == maxSize

    fun getLast(): T {
        require(size > 0) { "Buffer is empty" }
        @Suppress("UNCHECKED_CAST")
        return buffer[(start + size - 1) % maxSize] as T
    }
}

fun Point.add(other: Point): Point {
    return Point(this.x + other.x, this.y + other.y)
}

fun Point.subtract(other: Point): Point {
    return Point(this.x - other.x, this.y - other.y)
}

fun Point.distanceTo(other: Point): Double {
    return sqrt((this.x - other.x).pow(2.0) + (this.y - other.y).pow(2.0))
}

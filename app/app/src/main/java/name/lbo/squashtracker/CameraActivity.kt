package name.lbo.squashtracker

// Kotlin example demonstrating Camera2 API setup for 720p@60fps using the wide-angle lens,
// capturing frames to both MediaRecorder (for video) and ImageReader (for OpenCV processing),
// integrated into Jetpack Compose via AndroidView.
// Designed for minSdkVersion 27+.

import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.media.ImageReader
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.Surface
import android.view.TextureView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.annotation.RequiresApi
import androidx.annotation.RequiresPermission
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File

class Camera2ProcessorActivity : ComponentActivity() {
    private lateinit var cameraManager: CameraManager
    private var cameraDevice: CameraDevice? = null
    private lateinit var recorder: MediaRecorder
    private var session: CameraCaptureSession? = null
    private lateinit var backgroundThread: HandlerThread
    private lateinit var backgroundHandler: Handler
    private lateinit var reader: ImageReader

    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        startBackgroundThread()
        setContent {
            CameraPreviewAndProcessor()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        session?.close()
        cameraDevice?.close()
        reader.close()
        stopBackgroundThread()
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread.quitSafely()
        backgroundThread.join()
    }

    @Composable
    fun CameraPreviewAndProcessor() {
        Box {
            AndroidView(factory = { ctx ->
                TextureView(ctx).apply {
                    surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                        @RequiresPermission(Manifest.permission.CAMERA)
                        override fun onSurfaceTextureAvailable(
                            surface: SurfaceTexture,
                            width: Int,
                            height: Int
                        ) {
                            setupCameraAndSession(surface, width, height)
                        }

                        override fun onSurfaceTextureSizeChanged(
                            surface: SurfaceTexture,
                            width: Int,
                            height: Int
                        ) {
                        }

                        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean =
                            true

                        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
                    }
                }
            }, modifier = Modifier.matchParentSize())
        }
    }

    @RequiresApi(Build.VERSION_CODES.S)
    @SuppressLint("MissingPermission")
    private fun setupCameraAndSession(
        surfaceTexture: SurfaceTexture,
        width: Int,
        height: Int
    ) {
        // 1) Find wide-angle camera ID by smallest focal length
        val wideId = cameraManager.cameraIdList.first { id ->
            cameraManager.getCameraCharacteristics(id).let { chars ->
                chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK &&
                        (chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                            ?.minOrNull() ?: Float.MAX_VALUE) ==
                        (chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                            ?.minOrNull() ?: Float.MAX_VALUE)
            }
        }

        val mediaSize = Size(1280, 720)

        // 2) Configure MediaRecorder with context-aware constructor
        val recorder = MediaRecorder(this@Camera2ProcessorActivity).apply {
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoSize(mediaSize.width, mediaSize.height)
            setVideoFrameRate(60)
            setOutputFile(File(filesDir, "output.mp4").absolutePath)
            prepare()
        }

        // 3) ImageReader for OpenCV
        reader = ImageReader.newInstance(
            mediaSize.width,
            mediaSize.height,
            ImageFormat.YUV_420_888,
            2
        ).apply {
            setOnImageAvailableListener({ rdr ->
                rdr.acquireNextImage().use { image ->
                    val y = image.planes[0].buffer
                    val u = image.planes[1].buffer
                    val v = image.planes[2].buffer
                    val data = ByteArray(y.remaining() + u.remaining() + v.remaining()).also {
                        y.get(it, 0, y.remaining())
                        u.get(it, y.remaining(), u.remaining())
                        v.get(it, y.remaining() + u.remaining(), v.remaining())
                    }
                    val matYuv = Mat(
                        mediaSize.height + mediaSize.height / 2,
                        mediaSize.width,
                        CvType.CV_8UC1
                    )
                    matYuv.put(0, 0, data)
                    val rgb = Mat()
                    Imgproc.cvtColor(matYuv, rgb, Imgproc.COLOR_YUV2RGB_I420)
                    processFrame(rgb)
                    matYuv.release()
                    rgb.release()
                }
            }, backgroundHandler)
        }

        // 4) Open camera and create session with SessionConfiguration for high-speed capture
        cameraManager.openCamera(wideId, object : CameraDevice.StateCallback() {
            @RequiresApi(Build.VERSION_CODES.P)
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                surfaceTexture.setDefaultBufferSize(mediaSize.width, mediaSize.height)
                val previewSurface = Surface(surfaceTexture)
                val recordSurface = recorder.surface
                val processSurface = reader.surface

                val targets = listOf(previewSurface, recordSurface, processSurface)
                val executor = HandlerExecutor(backgroundHandler.looper)
                val outputConfigs = targets.map { OutputConfiguration(it) }
                val sessionConfig = SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    outputConfigs,
                    executor,
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(sessionBase: CameraCaptureSession) {
                            session = sessionBase
                            // Build high-speed capture request
                            val builder =
                                device.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
                                    targets.forEach { addTarget(it) }
                                    val characs = cameraManager.getCameraCharacteristics(device.id)
                                    val map =
                                        characs.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
                                    val ranges =
                                        map.getHighSpeedVideoFpsRangesFor(mediaSize).first()
                                    set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, ranges)
                                }
                            val burstList =
                                (sessionBase as CameraConstrainedHighSpeedCaptureSession)
                                    .createHighSpeedRequestList(builder.build())
                            sessionBase.setRepeatingBurst(burstList, null, backgroundHandler)
                            recorder.start()
                        }

                        override fun onConfigureFailed(session: CameraCaptureSession) {}
                    }
                )
                device.createCaptureSession(sessionConfig)
            }

            override fun onDisconnected(device: CameraDevice) = device.close()
            override fun onError(device: CameraDevice, error: Int) = device.close()
        }, backgroundHandler)
    }

    private fun processFrame(mat: Mat) {
        // Example: simple Canny edge detection
        val edges = Mat()
        Imgproc.Canny(mat, edges, 50.0, 150.0)
        // TODO: handle edges
        edges.release()
    }
}

// HandlerExecutor: wraps a Handler into an Executor for SessionConfiguration
class HandlerExecutor(private val handlerLooper: android.os.Looper) :
    java.util.concurrent.Executor {
    override fun execute(command: Runnable) {
        android.os.Handler(handlerLooper).post(command)
    }
}

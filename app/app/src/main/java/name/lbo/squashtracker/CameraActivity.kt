package name.lbo.squashtracker


import android.Manifest
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.media.MediaRecorder
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.Surface
import android.view.TextureView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
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
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        startBackgroundThread()
        setContent {
            CameraPreviewAndProcessor()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopBackgroundThread()
        cameraDevice?.close()
        reader.close()
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
        var textureRef by remember { mutableStateOf<TextureView?>(null) }
        Box {
            AndroidView(
                factory = { ctx ->
                    TextureView(ctx).apply {
                        surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                            override fun onSurfaceTextureSizeChanged(?, w: Int, h: Int) {}
                            override fun onSurfaceTextureUpdated(?) {}
                            override fun onSurfaceTextureDestroyed(?) = true
                            override fun onSurfaceTextureAvailable(st, w: Int, h: Int) {
                                setupCameraAndSession(st, w, h)
                            }
                        }
                    }.also { textureRef = it }
                },
                modifier = Modifier.matchParentSize()
            )
        }
    }

    @RequiresPermission(Manifest.permission.CAMERA)
    private fun setupCameraAndSession(
        surfaceTexture: android.graphics.SurfaceTexture,
        width: Int,
        height: Int
    ) {
        // 1) Find wide-angle camera ID by selecting lens with smallest focal length
        val wideId = cameraManager.cameraIdList.first { id ->
            cameraManager.getCameraCharacteristics(id).let { chars ->
                chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK &&
                        chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                            ?.minOrNull() == chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    ?.minOrNull()
            }
        }

        // 2) Choose 720p@60fps high-speed
        val mediaSize = Size(1280, 720)
        recorder = MediaRecorder().apply {
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoSize(mediaSize.width, mediaSize.height)
            setVideoFrameRate(60)
            setOutputFile(File(filesDir, "output.mp4").absolutePath)
            prepare()
        }

        // 3) ImageReader for OpenCV processing
        reader = ImageReader.newInstance(
            mediaSize.width, mediaSize.height,
            ImageFormat.YUV_420_888, 2
        ).apply {
            setOnImageAvailableListener({ reader ->
                reader.acquireNextImage().use { image ->
                    // convert to Mat
                    val yBuffer = image.planes[0].buffer
                    val uBuffer = image.planes[1].buffer
                    val vBuffer = image.planes[2].buffer
                    val ySize = yBuffer.remaining()
                    val uSize = uBuffer.remaining()
                    val vSize = vBuffer.remaining()
                    val data = ByteArray(ySize + uSize + vSize)
                    yBuffer.get(data, 0, ySize)
                    uBuffer.get(data, ySize, uSize)
                    vBuffer.get(data, ySize + uSize, vSize)
                    val mat = Mat(
                        mediaSize.height + mediaSize.height / 2,
                        mediaSize.width, CvType.CV_8UC1
                    )
                    mat.put(0, 0, data)
                    val rgb = Mat()
                    Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_YUV2RGB_I420)
                    processFrame(rgb)
                    mat.release(); rgb.release()
                }
            }, backgroundHandler)
        }

        // 4) Open camera and create session
        cameraManager.openCamera(wideId, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                val textureSurface = Surface(surfaceTexture.apply {
                    setDefaultBufferSize(mediaSize.width, mediaSize.height)
                })
                val recordingSurface = recorder.surface
                val processingSurface = reader.surface
                device.createCaptureSession(
                    listOf(recordingSurface, processingSurface, textureSurface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            this@Camera2ProcessorActivity.session = session
                            val builder = device.createCaptureRequest(
                                CameraDevice.TEMPLATE_RECORD
                            )
                            builder.addTarget(recordingSurface)
                            builder.addTarget(processingSurface)
                            builder.addTarget(textureSurface)
                            // set high-speed fps range
                            val ranges = device.cameraCharacteristics
                                .get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
                                .getHighSpeedVideoFpsRangesFor(mediaSize).first()
                            builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, ranges)
                            session.setRepeatingBurst(
                                CameraConstrainedHighSpeedCaptureSession
                                    .createHighSpeedRequestList(builder.build()),
                                null, backgroundHandler
                            )
                            recorder.start()
                        }

                        override fun onConfigureFailed(_) {}
                    }, backgroundHandler
                )
            }

            override fun onDisconnected(_) = device.close()
            override fun onError(_, __) = device.close()
        }, backgroundHandler)
    }

    private fun processFrame(mat: Mat) {
        // Example: simple Canny edge detection
        val edges = Mat()
        Imgproc.Canny(mat, edges, 50.0, 150.0)
        // TODO: display or analyze edges
        edges.release()
    }
}

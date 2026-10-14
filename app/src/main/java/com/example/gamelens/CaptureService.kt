package com.example.gamelens

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.view.Gravity
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager

/**
 * Foreground service that:
 *  1. holds the MediaProjection (capturing the game app, or the whole screen),
 *  2. shows a click-through full-screen overlay (a SurfaceView),
 *  3. feeds captured frames through [GlPipeline] and presents the enhanced result on the overlay.
 *
 * Tip: when the system dialog asks what to share, choose "A single app" and pick your game.
 * Single-app capture does not include other apps' windows, so our own overlay is not captured
 * back into itself (which would cause a feedback loop in whole-screen mode).
 */
class CaptureService : Service() {

    private lateinit var windowManager: WindowManager
    private lateinit var displayManager: DisplayManager
    private val main = Handler(Looper.getMainLooper())

    private var projection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var pipeline: GlPipeline? = null
    private var overlay: SurfaceView? = null
    private var overlayWm: WindowManager? = null
    private var dpi = 420

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}
        override fun onDisplayRemoved(displayId: Int) {}
        override fun onDisplayChanged(displayId: Int) {
            if (displayId != android.view.Display.DEFAULT_DISPLAY) return
            val (w, h) = realSize()
            val vd = virtualDisplay ?: return
            vd.resize(w, h, dpi)
            pipeline?.resize(w, h)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        displayManager = getSystemService(DISPLAY_SERVICE) as DisplayManager
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (projection != null) return START_NOT_STICKY // already running

        startForegroundCompat()

        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, 0) ?: 0
        @Suppress("DEPRECATION")
        val data = intent?.getParcelableExtra<Intent>(EXTRA_DATA)
        if (data == null) {
            stopSelf()
            return START_NOT_STICKY
        }

        val mpm = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val proj = mpm.getMediaProjection(resultCode, data)
        projection = proj
        // Android 14+: a callback must be registered before creating the VirtualDisplay.
        proj.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                main.post { stopSelf() }
            }
        }, main)

        addOverlay()
        return START_NOT_STICKY
    }

    private fun addOverlay() {
        // Prefer the trusted accessibility overlay (touches pass through to the game). Fall back to a
        // normal "draw over other apps" overlay only if the accessibility service isn't running.
        val a11y = GameLensAccessibilityService.instance
        val wm = (a11y?.getSystemService(WINDOW_SERVICE) as? WindowManager) ?: windowManager
        val view = SurfaceView(a11y ?: this)
        overlay = view
        overlayWm = wm

        val lp = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            if (a11y != null) WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
            else WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            // Android blocks touches that pass through an untrusted overlay whose window alpha is
            // above ~0.8 (the "maximum obscuring opacity"). Stay just under it so taps reach the game.
            alpha = OVERLAY_ALPHA
            layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
        }

        view.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                startPipeline(holder)
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, w: Int, h: Int) {}

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                stopPipeline()
            }
        })

        wm.addView(view, lp)
    }

    private fun startPipeline(holder: SurfaceHolder) {
        val proj = projection ?: return
        val (w, h) = realSize()
        val p = GlPipeline(holder.surface, w, h)
        p.start()
        pipeline = p

        virtualDisplay = proj.createVirtualDisplay(
            "GameLensCapture",
            w, h, dpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            p.captureSurface,
            null, null,
        )
        displayManager.registerDisplayListener(displayListener, main)
    }

    private fun stopPipeline() {
        runCatching { displayManager.unregisterDisplayListener(displayListener) }
        virtualDisplay?.release()
        virtualDisplay = null
        pipeline?.release()
        pipeline = null
    }

    private fun realSize(): Pair<Int, Int> {
        val display = displayManager.getDisplay(android.view.Display.DEFAULT_DISPLAY)
        val m = DisplayMetrics()
        @Suppress("DEPRECATION")
        display.getRealMetrics(m)
        dpi = m.densityDpi
        return m.widthPixels to m.heightPixels
    }

    private fun startForegroundCompat() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "GameLens", NotificationManager.IMPORTANCE_LOW)
        )
        val stop = PendingIntent.getService(
            this, 0,
            Intent(this, CaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val n = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("GameLens is enhancing your screen")
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .setOngoing(true)
            .build()
        startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
    }

    override fun onDestroy() {
        stopPipeline()
        overlay?.let { runCatching { (overlayWm ?: windowManager).removeView(it) } }
        overlay = null
        overlayWm = null
        projection?.stop()
        projection = null
        running = false
        super.onDestroy()
    }

    companion object {
        const val ACTION_STOP = "com.example.gamelens.STOP"
        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_DATA = "data"
        private const val CHANNEL_ID = "gamelens"
        private const val OVERLAY_ALPHA = 0.79f

        @Volatile var running = false

        fun start(ctx: Context, resultCode: Int, data: Intent) {
            running = true
            ctx.startForegroundService(
                Intent(ctx, CaptureService::class.java)
                    .putExtra(EXTRA_RESULT_CODE, resultCode)
                    .putExtra(EXTRA_DATA, data)
            )
        }

        fun stop(ctx: Context) {
            ctx.startService(Intent(ctx, CaptureService::class.java).setAction(ACTION_STOP))
        }
    }
}

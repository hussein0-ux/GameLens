package com.example.gamelens

import android.Manifest
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts

/** Minimal programmatic UI: start/stop plus live look sliders. */
class MainActivity : ComponentActivity() {

    private val capturePermission =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
            val data = res.data
            if (res.resultCode == RESULT_OK && data != null) {
                CaptureService.start(this, res.resultCode, data)
                Toast.makeText(this, "Now open your game", Toast.LENGTH_LONG).show()
            }
        }

    private val notifPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        notifPermission.launch(Manifest.permission.POST_NOTIFICATIONS)

        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 2, pad, pad)
        }

        root.addView(TextView(this).apply {
            text = "GameLens"
            textSize = 26f
        })
        root.addView(TextView(this).apply {
            text = "Real-time sharpening, clarity and filmic grading for your game.\n" +
                "When Android asks what to share, choose \"A single app\" and pick your game."
        })

        root.addView(Button(this).apply {
            text = "Start"
            setOnClickListener { start() }
        })
        root.addView(Button(this).apply {
            text = "Stop"
            setOnClickListener { CaptureService.stop(this@MainActivity) }
        })

        slider(root, "Sharpen", 0f, 1f, LookSettings.sharpen) { LookSettings.sharpen = it }
        slider(root, "Clarity", 0f, 1f, LookSettings.clarity) { LookSettings.clarity = it }
        slider(root, "Contrast", 0.8f, 1.4f, LookSettings.contrast) { LookSettings.contrast = it }
        slider(root, "Saturation", 0.5f, 1.6f, LookSettings.saturation) { LookSettings.saturation = it }
        slider(root, "Filmic", 0f, 1f, LookSettings.filmic) { LookSettings.filmic = it }

        setContentView(root)
    }

    private fun start() {
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Allow \"Display over other apps\", then press Start again", Toast.LENGTH_LONG).show()
            startActivity(
                Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName"))
            )
            return
        }
        val mpm = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        capturePermission.launch(mpm.createScreenCaptureIntent())
    }

    private fun slider(
        parent: LinearLayout,
        label: String,
        min: Float,
        max: Float,
        initial: Float,
        onChange: (Float) -> Unit,
    ) {
        val steps = 100
        val title = TextView(this).apply { text = "$label: %.2f".format(initial) }
        val bar = SeekBar(this).apply {
            this.max = steps
            progress = (((initial - min) / (max - min)) * steps).toInt()
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                    val v = min + (max - min) * p / steps
                    title.text = "$label: %.2f".format(v)
                    onChange(v)
                }
                override fun onStartTrackingTouch(sb: SeekBar) {}
                override fun onStopTrackingTouch(sb: SeekBar) {}
            })
        }
        parent.addView(title)
        parent.addView(bar)
    }
}

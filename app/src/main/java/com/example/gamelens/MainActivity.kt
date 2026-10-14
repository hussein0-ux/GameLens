package com.example.gamelens

import android.Manifest
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.provider.Settings
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts

/** Minimal programmatic UI: start/stop, look presets and live look sliders. */
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

    /** Each slider registers a function that re-reads its value (used after a preset is applied). */
    private val refreshers = mutableListOf<() -> Unit>()

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
            text = "Build 0.4 - AI upscaler benchmark"
            textSize = 12f
        })
        root.addView(TextView(this).apply {
            text = "Real-time sharpening, grading and glow for your game.\n" +
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

        root.addView(Button(this).apply {
            text = "AI upscaler benchmark"
            setOnClickListener { startActivity(Intent(this@MainActivity, BenchmarkActivity::class.java)) }
        })

        val presets = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        fun presetButton(label: String, p: LookSettings.Preset) {
            presets.addView(
                Button(this).apply {
                    text = label
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    setOnClickListener {
                        LookSettings.apply(p)
                        refreshers.forEach { it() }
                    }
                }
            )
        }
        presetButton("Natural", LookSettings.NATURAL)
        presetButton("Vivid", LookSettings.VIVID)
        presetButton("Cinematic", LookSettings.CINEMATIC)
        root.addView(presets)

        slider(root, "Sharpen", 0f, 1f, { LookSettings.sharpen }) { LookSettings.sharpen = it }
        slider(root, "Clarity", 0f, 1f, { LookSettings.clarity }) { LookSettings.clarity = it }
        slider(root, "Exposure", 0.8f, 1.4f, { LookSettings.exposure }) { LookSettings.exposure = it }
        slider(root, "Contrast", 0.8f, 1.4f, { LookSettings.contrast }) { LookSettings.contrast = it }
        slider(root, "Saturation", 0.5f, 1.6f, { LookSettings.saturation }) { LookSettings.saturation = it }
        slider(root, "Vibrance", 0f, 1f, { LookSettings.vibrance }) { LookSettings.vibrance = it }
        slider(root, "Tone curve", 0f, 1f, { LookSettings.filmic }) { LookSettings.filmic = it }
        slider(root, "Teal/orange grade", 0f, 1f, { LookSettings.grade }) { LookSettings.grade = it }
        slider(root, "Glow", 0f, 1f, { LookSettings.glow }) { LookSettings.glow = it }

        setContentView(ScrollView(this).apply { addView(root) })
    }

    private fun start() {
        if (GameLensAccessibilityService.instance == null) {
            Toast.makeText(
                this,
                "Turn on \"GameLens overlay\" under Installed apps, then press Start again. " +
                    "If it is greyed out: App info > ⋮ > Allow restricted settings.",
                Toast.LENGTH_LONG,
            ).show()
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
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
        get: () -> Float,
        set: (Float) -> Unit,
    ) {
        val steps = 100
        val title = TextView(this)
        val bar = SeekBar(this).apply {
            this.max = steps
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        fun show(v: Float) {
            title.text = "$label: %.2f".format(v)
        }

        fun refresh() {
            val v = get()
            show(v)
            bar.progress = (((v - min) / (max - min)) * steps).toInt().coerceIn(0, steps)
        }

        bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                if (!fromUser) return // programmatic refresh must not overwrite exact preset values
                val v = min + (max - min) * p / steps
                show(v)
                set(v)
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {}
        })

        refresh()
        refreshers.add { refresh() }
        parent.addView(title)
        parent.addView(bar)
    }
}

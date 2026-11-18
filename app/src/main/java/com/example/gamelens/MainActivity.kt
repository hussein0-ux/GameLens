package com.example.gamelens

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.view.View
import android.widget.CheckBox
import android.widget.RadioButton
import android.widget.RadioGroup
import java.io.File
import java.util.zip.ZipInputStream
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

    private val pickOnnx =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) loadModel(uri)
        }

    private val ui = Handler(Looper.getMainLooper())
    private var statsView: TextView? = null
    private var aiModelLabel: TextView? = null
    private val statsTick = object : Runnable {
        override fun run() {
            val st = AiSettings.stats
            val dg = AiSettings.diag
            statsView?.text = AiSettings.status + (if (st.isNotEmpty()) "\n$st" else "") + (if (dg.isNotEmpty()) "\n$dg" else "")
            ui.postDelayed(this, 500)
        }
    }

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
            text = "Build 0.5 - Crisp default, HDR look, crash log"
            textSize = 12f
        })
        root.addView(TextView(this).apply {
            text = "Real-time sharpening, grading and glow for your game.\n" +
                "When Android asks what to share, choose \"A single app\" and pick your game."
        })

        val crashFile = GameLensApp.crashFile(this)
        if (crashFile.exists() && crashFile.length() > 0) {
            val log = crashFile.readText()
            root.addView(TextView(this).apply {
                text = "GameLens recorded a problem last time:\n" + log.takeLast(1800)
                typeface = android.graphics.Typeface.MONOSPACE
                textSize = 10f
                setTextColor(0xFFFF6B6B.toInt())
            })
            root.addView(Button(this).apply {
                text = "Copy crash log, then clear it"
                setOnClickListener {
                    val cm = getSystemService(android.content.ClipboardManager::class.java)
                    cm.setPrimaryClip(android.content.ClipData.newPlainText("GameLens crash", log))
                    crashFile.delete()
                    Toast.makeText(this@MainActivity, "Copied. Paste it to Claude.", Toast.LENGTH_LONG).show()
                }
            })
        }

        root.addView(Button(this).apply {
            text = "Start"
            setOnClickListener { start() }
        })
        root.addView(Button(this).apply {
            text = "Stop"
            setOnClickListener { CaptureService.stop(this@MainActivity) }
        })

        addAiSection(root, pad)

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
        presetButton("Montage", LookSettings.MONTAGE)
        root.addView(presets)
        val presets2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        presets2.addView(Button(this).apply {
            text = "HDR Pop"
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { LookSettings.apply(LookSettings.HDR_POP); refreshers.forEach { it() } }
        })
        presets2.addView(Button(this).apply {
            text = "Crisp"
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { LookSettings.apply(LookSettings.CRISP); refreshers.forEach { it() } }
        })
        root.addView(presets2)

        root.addView(TextView(this).apply { text = "Mode" })
        radioGroup(
            root,
            listOf(
                "Look filters only (heavy, no upscale)",
                "GPU upscale + LUT (fast)",
                "Neural upscale + LUT (needs a model and a 1170x540 game)",
                "Auto: neural, falls back to GPU if it cannot hold the frame rate",
            ),
            LookSettings.mode,
        ) { i -> LookSettings.mode = i }
        root.addView(CheckBox(this).apply {
            text = "Capture at half screen size (applies on next start)"
            isChecked = LookSettings.captureHalf
            setOnCheckedChangeListener { _, c -> LookSettings.captureHalf = c }
        })
        root.addView(TextView(this).apply { text = "GPU upscaler" })
        radioGroup(
            root,
            listOf("Qualcomm GSR (edge-directed, made for games)", "Catmull-Rom + sharpen"),
            LookSettings.upscaler,
        ) { i -> LookSettings.upscaler = i }
        slider(root, "Upscale sharpness", 0f, 1f, { LookSettings.upSharp }) { LookSettings.upSharp = it }
        slider(root, "Edge anti-aliasing (GSR only)", 0f, 1f, { LookSettings.aa }) { LookSettings.aa = it }
        slider(root, "Enhance clarity (GSR only)", 0f, 1f, { LookSettings.upClarity }) { LookSettings.upClarity = it }
        slider(root, "Enhance vibrance (GSR only)", 0f, 1f, { LookSettings.upVibrance }) { LookSettings.upVibrance = it }
        root.addView(CheckBox(this).apply {
            text = "Bypass look (raw capture-rate test)"
            isChecked = LookSettings.bypass
            setOnCheckedChangeListener { _, c -> LookSettings.bypass = c }
        })
        slider(root, "LUT strength (HyperReal)", 0f, 1f, { LookSettings.lut }) { LookSettings.lut = it }
        slider(root, "Sharpen", 0f, 1f, { LookSettings.sharpen }) { LookSettings.sharpen = it }
        slider(root, "Clarity", 0f, 1f, { LookSettings.clarity }) { LookSettings.clarity = it }
        slider(root, "Exposure", 0.8f, 1.4f, { LookSettings.exposure }) { LookSettings.exposure = it }
        slider(root, "Contrast", 0.8f, 1.4f, { LookSettings.contrast }) { LookSettings.contrast = it }
        slider(root, "Saturation", 0.5f, 1.6f, { LookSettings.saturation }) { LookSettings.saturation = it }
        slider(root, "Vibrance", 0f, 1f, { LookSettings.vibrance }) { LookSettings.vibrance = it }
        slider(root, "Tone curve", 0f, 1f, { LookSettings.filmic }) { LookSettings.filmic = it }
        slider(root, "Teal/orange grade", 0f, 1f, { LookSettings.grade }) { LookSettings.grade = it }
        slider(root, "HDR look", 0f, 1f, { LookSettings.hdr }) { LookSettings.hdr = it }
        slider(root, "Deep blacks", 0f, 1f, { LookSettings.blackPt }) { LookSettings.blackPt = it }
        slider(root, "Glow", 0f, 1f, { LookSettings.glow }) { LookSettings.glow = it }
        slider(root, "Vignette", 0f, 1f, { LookSettings.vignette }) { LookSettings.vignette = it }
        slider(root, "Chromatic fringe", 0f, 1f, { LookSettings.chroma }) { LookSettings.chroma = it }
        slider(root, "Motion blur", 0f, 0.6f, { LookSettings.motion }) { LookSettings.motion = it }

        setContentView(ScrollView(this).apply { addView(root) })
    }

    override fun onResume() {
        super.onResume()
        ui.post(statsTick)
    }

    override fun onPause() {
        ui.removeCallbacks(statsTick)
        super.onPause()
    }

    private fun addAiSection(root: LinearLayout, pad: Int) {
        AiSettings.nativeLibDir = applicationInfo.nativeLibraryDir
        getSharedPreferences("ai", MODE_PRIVATE).getString("model", null)?.let {
            if (it.endsWith(".tflite") && File(it).exists()) AiSettings.modelPath = it
        }

        root.addView(TextView(this).apply {
            text = "Neural model (Snapdragon NPU)"
            textSize = 18f
            setPadding(0, pad, 0, 0)
        })
        root.addView(TextView(this).apply {
            text = "Pick the int8 planar 2x .tflite from AI Hub (.tflite or the .zip). Its input size must match the game's " +
                "render size, for example 1168x540 for a game rendering at 1170x540. Then choose Neural or Auto above."
            textSize = 12f
        })
        root.addView(Button(this).apply {
            text = "Pick neural model (.tflite / .zip)"
            setOnClickListener { pickOnnx.launch(arrayOf("*/*")) }
        })
        aiModelLabel = TextView(this).apply {
            text = "Model: " + (AiSettings.modelPath?.let { File(it).name } ?: "(none)")
        }
        root.addView(aiModelLabel)

        statsView = TextView(this).apply {
            typeface = android.graphics.Typeface.MONOSPACE
            textSize = 12f
        }
        root.addView(statsView)
    }

    private fun radioGroup(parent: LinearLayout, labels: List<String>, selected: Int, onPick: (Int) -> Unit) {
        val g = RadioGroup(this)
        labels.forEachIndexed { i, l ->
            g.addView(RadioButton(this).apply {
                text = l
                id = View.generateViewId()
                isChecked = i == selected
            })
        }
        g.setOnCheckedChangeListener { grp, id -> onPick(grp.indexOfChild(grp.findViewById(id))) }
        parent.addView(g)
    }

    /** Accepts a .tflite or a .zip containing one. The file is kept in app storage so the choice survives restarts. */
    private fun loadModel(uri: Uri) {
        try {
            val bytes = contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            var tflite: ByteArray? = null
            var name = "model.tflite"
            if (bytes.size > 4 && bytes[0] == 'P'.code.toByte() && bytes[1] == 'K'.code.toByte()) {
                ZipInputStream(bytes.inputStream()).use { zin ->
                    var e = zin.nextEntry
                    while (e != null && tflite == null) {
                        if (!e.isDirectory && e.name.endsWith(".tflite")) {
                            tflite = zin.readBytes()
                            name = File(e.name).name
                        }
                        e = zin.nextEntry
                    }
                }
            } else {
                tflite = bytes
            }
            val data = tflite ?: throw IllegalStateException("No .tflite file found in the zip")
            val dir = File(filesDir, "neural_model").apply { deleteRecursively(); mkdirs() }
            val f = File(dir, name).also { it.writeBytes(data) }
            AiSettings.modelPath = f.path
            AiSettings.generation++
            getSharedPreferences("ai", MODE_PRIVATE).edit().putString("model", f.path).apply()
            aiModelLabel?.text = "Model: ${f.name}"
        } catch (t: Throwable) {
            Toast.makeText(this, "Could not load model: ${t.message}", Toast.LENGTH_LONG).show()
        }
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

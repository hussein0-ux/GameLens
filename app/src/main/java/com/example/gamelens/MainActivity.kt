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
            if (uri != null) loadOnnx(uri)
        }

    private val ui = Handler(Looper.getMainLooper())
    private var statsView: TextView? = null
    private var aiModelLabel: TextView? = null
    private val statsTick = object : Runnable {
        override fun run() {
            val st = AiSettings.stats
            statsView?.text = AiSettings.status + if (st.isNotEmpty()) "\n$st" else ""
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
            text = "Build 0.3 - trusted overlay, grade + glow"
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

        slider(root, "Sharpen", 0f, 1f, { LookSettings.sharpen }) { LookSettings.sharpen = it }
        slider(root, "Clarity", 0f, 1f, { LookSettings.clarity }) { LookSettings.clarity = it }
        slider(root, "Exposure", 0.8f, 1.4f, { LookSettings.exposure }) { LookSettings.exposure = it }
        slider(root, "Contrast", 0.8f, 1.4f, { LookSettings.contrast }) { LookSettings.contrast = it }
        slider(root, "Saturation", 0.5f, 1.6f, { LookSettings.saturation }) { LookSettings.saturation = it }
        slider(root, "Vibrance", 0f, 1f, { LookSettings.vibrance }) { LookSettings.vibrance = it }
        slider(root, "Tone curve", 0f, 1f, { LookSettings.filmic }) { LookSettings.filmic = it }
        slider(root, "Teal/orange grade", 0f, 1f, { LookSettings.grade }) { LookSettings.grade = it }
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
            if (File(it).exists()) AiSettings.modelPath = it
        }

        root.addView(TextView(this).apply {
            text = "AI upscale (experimental, Snapdragon only)"
            textSize = 18f
            setPadding(0, pad, 0, 0)
        })
        root.addView(TextView(this).apply {
            text = "Shrinks the screen to a small image, upscales it 4x on the NPU and shows that instead of " +
                "the look filters below. Use landscape games. Start the capture, then watch the stats here."
            textSize = 12f
        })
        root.addView(CheckBox(this).apply {
            text = "Enable AI upscale"
            isChecked = AiSettings.enabled
            setOnCheckedChangeListener { _, c ->
                AiSettings.enabled = c
                AiSettings.generation++
            }
        })
        root.addView(Button(this).apply {
            text = "Pick ONNX model (.zip / .onnx)"
            setOnClickListener { pickOnnx.launch(arrayOf("*/*")) }
        })
        aiModelLabel = TextView(this).apply {
            text = "Model: " + (AiSettings.modelPath?.let { File(it).name } ?: "(none)")
        }
        root.addView(aiModelLabel)

        val grids = listOf(4 to 2, 6 to 3, 7 to 3)
        radioGroup(
            root,
            listOf("Input 512x256 (8 tiles, fastest)", "Input 768x384 (18 tiles)", "Input 896x384 (21 tiles)"),
            0,
        ) { i -> AiSettings.cols = grids[i].first; AiSettings.rows = grids[i].second }
        radioGroup(
            root,
            listOf("AI only", "Split: AI left, plain bilinear right", "Bilinear only"),
            AiSettings.viewMode,
        ) { i -> AiSettings.viewMode = i }

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

    /** Accepts a .onnx or a .zip containing one. Files are kept in app storage so the choice survives restarts. */
    private fun loadOnnx(uri: Uri) {
        try {
            val bytes = contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            val name = contentResolver.query(uri, null, null, null, null)?.use { c ->
                val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (i >= 0 && c.moveToFirst()) c.getString(i) else null
            } ?: "model.onnx"
            val dir = File(filesDir, "onnx_model").apply { deleteRecursively(); mkdirs() }
            var onnx: File? = null
            if (bytes.size > 4 && bytes[0] == 'P'.code.toByte() && bytes[1] == 'K'.code.toByte()) {
                ZipInputStream(bytes.inputStream()).use { zin ->
                    var e = zin.nextEntry
                    while (e != null) {
                        val base = File(e.name).name
                        if (!e.isDirectory && (base.endsWith(".onnx") || base.endsWith(".data"))) {
                            val f = File(dir, base)
                            f.writeBytes(zin.readBytes())
                            if (base.endsWith(".onnx")) onnx = f
                        }
                        e = zin.nextEntry
                    }
                }
            } else if (name.endsWith(".onnx")) {
                onnx = File(dir, name).also { it.writeBytes(bytes) }
            }
            val f = onnx ?: throw IllegalStateException("No .onnx file found in $name")
            // Models with external weights reference a sidecar file (e.g. "model.data") by name inside the .onnx.
            // If it is missing, QNN rejects every Conv with a confusing NHWC error, so fail early with a clear message.
            val text = String(f.readBytes(), Charsets.ISO_8859_1)
            val missing = Regex("[A-Za-z0-9_.\\-]+\\.(data|bin|weights)").findAll(text)
                .map { it.value }.distinct().filter { !File(dir, it).exists() }.toList()
            if (missing.isNotEmpty()) {
                f.delete()
                throw IllegalStateException(
                    "Model needs external weights file(s) ${missing.joinToString()} that were not provided. " +
                        "Pick the .zip containing them, or use the single-file (inline) .onnx."
                )
            }
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

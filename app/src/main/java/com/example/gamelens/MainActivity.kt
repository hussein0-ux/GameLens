package com.example.gamelens

import android.Manifest
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File
import java.util.Locale
import java.util.zip.ZipInputStream

/** Minimal programmatic UI: start/stop, Snapdragon 8 Gen 3 120Hz HyperReal presets, and live look sliders. */
class MainActivity : ComponentActivity() {

    private val capturePermission =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
            val data = res.data
            if (res.resultCode == RESULT_OK && data != null) {
                CaptureService.start(this, res.resultCode, data)
                Toast.makeText(this, "Now open Call of Duty: Mobile", Toast.LENGTH_LONG).show()
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
    private var budgetView: TextView? = null
    private var aiModelLabel: TextView? = null
    private var modeGroup: RadioGroup? = null
    private var upscalerGroup: RadioGroup? = null
    private var captureHalfBox: CheckBox? = null
    private var lock120Box: CheckBox? = null
    private var bypassBox: CheckBox? = null

    private val statsTick = object : Runnable {
        override fun run() {
            val st = AiSettings.stats
            val dg = AiSettings.diag
            statsView?.text = AiSettings.status + (if (st.isNotEmpty()) "\n$st" else "") + (if (dg.isNotEmpty()) "\n$dg" else "")
            updateBudget()
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
            text = "Build 0.6 - Snapdragon 8 Gen 3 (Adreno 750) 120Hz HyperReal"
            textSize = 12f
        })
        root.addView(TextView(this).apply {
            text = "Real-time Qualcomm GSR upscale, micro-relief, dehaze, HDR tone mapping & LUT.\n" +
                "For CoD Mobile 120fps: set in-game Frame Rate = Ultra, choose \"A single app\" when sharing."
        })

        budgetView = TextView(this).apply {
            typeface = android.graphics.Typeface.MONOSPACE
            textSize = 11f
            setPadding(0, pad / 2, 0, pad / 2)
        }
        root.addView(budgetView)

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
                    Toast.makeText(this@MainActivity, "Copied crash log.", Toast.LENGTH_LONG).show()
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
        root.addView(Button(this).apply {
            text = "SD 8 Gen 3 · CoD Mobile 120Hz Extreme"
            setOnClickListener {
                LookSettings.applySd8Gen3Hyper120()
                syncAllControls()
                Toast.makeText(this@MainActivity, "8 Gen 3 120Hz Extreme HyperReal active", Toast.LENGTH_SHORT).show()
            }
        })

        addAiSection(root, pad)

        val presets = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        fun presetButton(row: LinearLayout, label: String, p: LookSettings.Preset) {
            row.addView(
                Button(this).apply {
                    text = label
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    setOnClickListener {
                        LookSettings.apply(p)
                        refreshers.forEach { it() }
                        updateBudget()
                    }
                }
            )
        }
        presetButton(presets, "Natural", LookSettings.NATURAL)
        presetButton(presets, "Vivid", LookSettings.VIVID)
        presetButton(presets, "Cinematic", LookSettings.CINEMATIC)
        presetButton(presets, "Montage", LookSettings.MONTAGE)
        root.addView(presets)

        val presets2 = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        presetButton(presets2, "HDR Pop", LookSettings.HDR_POP)
        presetButton(presets2, "Crisp", LookSettings.CRISP)
        presetButton(presets2, "8 Gen 3 Extreme", LookSettings.SD8_GEN3_HYPER)
        root.addView(presets2)

        root.addView(TextView(this).apply { text = "Mode" })
        modeGroup = radioGroup(
            root,
            listOf(
                "Look filters only (heavy, no upscale)",
                "GPU upscale + fused HyperReal + LUT (120Hz)",
                "Neural upscale + LUT (needs model matching capture size)",
                "Auto: neural, falls back to GPU if it drops below 110fps",
            ),
            LookSettings.mode,
        ) { i ->
            LookSettings.mode = i
            updateBudget()
        }
        captureHalfBox = CheckBox(this).apply {
            text = "Capture at half screen size (CoD Mobile 120Hz, next start)"
            isChecked = LookSettings.captureHalf
            setOnCheckedChangeListener { _, c ->
                LookSettings.captureHalf = c
                updateBudget()
            }
        }
        root.addView(captureHalfBox)
        lock120Box = CheckBox(this).apply {
            text = "Lock overlay surface to 120 Hz (next start)"
            isChecked = LookSettings.target120Hz
            setOnCheckedChangeListener { _, c -> LookSettings.target120Hz = c }
        }
        root.addView(lock120Box)

        root.addView(TextView(this).apply { text = "GPU upscaler" })
        upscalerGroup = radioGroup(
            root,
            listOf("Catmull-Rom + CAS + HyperReal", "Qualcomm GSR (edge-directed, made for games)"),
            LookSettings.upscaler,
        ) { i ->
            LookSettings.upscaler = i
            updateBudget()
        }
        slider(root, "Upscale sharpness", 0f, 1f, { LookSettings.upSharp }) { LookSettings.upSharp = it }
        slider(root, "Edge anti-aliasing (GSR pre-pass)", 0f, 1f, { LookSettings.aa }) { LookSettings.aa = it; updateBudget() }
        slider(root, "Upscale clarity (GPU mode)", 0f, 1f, { LookSettings.upClarity }) { LookSettings.upClarity = it; updateBudget() }
        slider(root, "Upscale vibrance (GPU mode)", 0f, 1f, { LookSettings.upVibrance }) { LookSettings.upVibrance = it }
        bypassBox = CheckBox(this).apply {
            text = "Bypass look (raw capture-rate test)"
            isChecked = LookSettings.bypass
            setOnCheckedChangeListener { _, c -> LookSettings.bypass = c }
        }
        root.addView(bypassBox)

        slider(root, "LUT strength (HyperReal)", 0f, 1f, { LookSettings.lut }) { LookSettings.lut = it }
        slider(root, "Micro-texture relief (8 Gen 3)", 0f, 1f, { LookSettings.microRelief }) { LookSettings.microRelief = it; updateBudget() }
        slider(root, "Atmospheric dehaze", 0f, 1f, { LookSettings.dehaze }) { LookSettings.dehaze = it; updateBudget() }
        slider(root, "Shadow detail recovery", 0f, 1f, { LookSettings.shadowLift }) { LookSettings.shadowLift = it; updateBudget() }
        slider(root, "HDR local tone & punch", 0f, 1f, { LookSettings.hdr }) { LookSettings.hdr = it; updateBudget() }
        slider(root, "ACES Filmic tone curve", 0f, 1f, { LookSettings.filmic }) { LookSettings.filmic = it }
        slider(root, "Exposure", 0.8f, 1.4f, { LookSettings.exposure }) { LookSettings.exposure = it }
        slider(root, "Contrast", 0.8f, 1.4f, { LookSettings.contrast }) { LookSettings.contrast = it }
        slider(root, "Saturation", 0.5f, 1.6f, { LookSettings.saturation }) { LookSettings.saturation = it }
        slider(root, "Teal/orange grade", 0f, 1f, { LookSettings.grade }) { LookSettings.grade = it }
        slider(root, "Deep blacks", 0f, 1f, { LookSettings.blackPt }) { LookSettings.blackPt = it }
        slider(root, "Specular glow / bloom", 0f, 1f, { LookSettings.glow }) { LookSettings.glow = it; updateBudget() }
        slider(root, "Vignette", 0f, 1f, { LookSettings.vignette }) { LookSettings.vignette = it }
        slider(root, "Sharpen (Look mode)", 0f, 1f, { LookSettings.sharpen }) { LookSettings.sharpen = it }
        slider(root, "Clarity (Look mode)", 0f, 1f, { LookSettings.clarity }) { LookSettings.clarity = it }
        slider(root, "Vibrance (Look mode)", 0f, 1f, { LookSettings.vibrance }) { LookSettings.vibrance = it }
        slider(root, "Chromatic fringe (Look mode)", 0f, 1f, { LookSettings.chroma }) { LookSettings.chroma = it }
        slider(root, "Motion blur (Look mode)", 0f, 0.6f, { LookSettings.motion }) { LookSettings.motion = it; updateBudget() }

        updateBudget()
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

    private fun syncAllControls() {
        (modeGroup?.getChildAt(LookSettings.mode) as? RadioButton)?.isChecked = true
        (upscalerGroup?.getChildAt(LookSettings.upscaler) as? RadioButton)?.isChecked = true
        captureHalfBox?.isChecked = LookSettings.captureHalf
        lock120Box?.isChecked = LookSettings.target120Hz
        bypassBox?.isChecked = LookSettings.bypass
        refreshers.forEach { it() }
        updateBudget()
    }

    private fun updateBudget() {
        val est = Sd8Gen3Profile.estimate()
        val label = if (est.safe120Hz) {
            setBudgetColor(0xFF4ADE80.toInt())
            String.format(
                Locale.US,
                "Adreno 750 @ 120Hz (8.33 ms window): est %.2f ms · SAFE 120 FPS (%.2f ms headroom)",
                est.gpuMs, est.headroomMs,
            )
        } else {
            setBudgetColor(0xFFFBBF24.toInt())
            String.format(
                Locale.US,
                "Adreno 750 @ 120Hz: est %.2f ms (> 4.50 ms target) — tap SD 8 Gen 3 button for 120Hz lock",
                est.gpuMs,
            )
        }
        budgetView?.text = label
    }

    private fun setBudgetColor(color: Int) {
        budgetView?.setTextColor(color)
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
            text = "Optional: int8 planar 2x .tflite from AI Hub matching game render size. " +
                "For guaranteed 120fps in CoD Mobile, GPU GSR + HyperReal above is faster and sharper."
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

    private fun radioGroup(parent: LinearLayout, labels: List<String>, selected: Int, onPick: (Int) -> Unit): RadioGroup {
        val g = RadioGroup(this)
        labels.forEachIndexed { i, l ->
            g.addView(RadioButton(this).apply {
                text = l
                id = View.generateViewId()
                isChecked = i == selected
            })
        }
        g.setOnCheckedChangeListener { grp, id ->
            val idx = grp.indexOfChild(grp.findViewById(id))
            if (idx >= 0) onPick(idx)
        }
        parent.addView(g)
        return g
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

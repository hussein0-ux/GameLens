package com.example.gamelens

/**
 * A 3D colour lookup table from an Adobe/Resolve ".cube" file.
 * [rgb] holds size^3 entries of 3 floats each, red index varying fastest (the order the file uses),
 * normalised to 0..1 by the file's DOMAIN_MIN / DOMAIN_MAX.
 */
class CubeLut(val size: Int, val rgb: FloatArray)

/** Parses the text of a .cube file. Throws IllegalArgumentException with a readable reason on bad input. */
fun parseCube(text: String): CubeLut {
    var size = 0
    var domainMin = floatArrayOf(0f, 0f, 0f)
    var domainMax = floatArrayOf(1f, 1f, 1f)
    val values = ArrayList<Float>()

    for (rawLine in text.lineSequence()) {
        val line = rawLine.trim()
        if (line.isEmpty() || line.startsWith("#") || line.startsWith("TITLE")) continue
        val parts = line.split(Regex("\\s+"))
        when (parts[0]) {
            "LUT_3D_SIZE" -> size = parts[1].toInt()
            "LUT_1D_SIZE" -> throw IllegalArgumentException("1D LUTs are not supported")
            "DOMAIN_MIN" -> domainMin = floatArrayOf(parts[1].toFloat(), parts[2].toFloat(), parts[3].toFloat())
            "DOMAIN_MAX" -> domainMax = floatArrayOf(parts[1].toFloat(), parts[2].toFloat(), parts[3].toFloat())
            else -> {
                require(parts.size == 3) { "Unexpected line in .cube file: $line" }
                for (c in 0 until 3) {
                    values.add((parts[c].toFloat() - domainMin[c]) / (domainMax[c] - domainMin[c]))
                }
            }
        }
    }

    require(size in 2..256) { "Missing or invalid LUT_3D_SIZE (got $size)" }
    val expected = size * size * size
    require(values.size == expected * 3) { "expected $expected entries, got ${values.size / 3}" }
    return CubeLut(size, values.toFloatArray())
}

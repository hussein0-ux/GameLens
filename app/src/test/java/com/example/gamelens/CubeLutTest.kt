package com.example.gamelens

import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class CubeLutTest {

    private val identity2 = """
        TITLE "identity"
        LUT_3D_SIZE 2
        0 0 0
        1 0 0
        0 1 0
        1 1 0
        0 0 1
        1 0 1
        0 1 1
        1 1 1
    """.trimIndent()

    @Test
    fun parsesSizeAndKeepsRedAsTheFastestAxis() {
        val lut = parseCube(identity2)
        assertEquals(2, lut.size)
        assertEquals(2 * 2 * 2 * 3, lut.rgb.size)
        assertEquals(listOf(1f, 0f, 0f), lut.rgb.slice(3..5))   // second entry: red index advanced
        assertEquals(listOf(0f, 1f, 0f), lut.rgb.slice(6..8))   // third entry: green index advanced
        assertEquals(listOf(1f, 1f, 1f), lut.rgb.slice(21..23))
    }

    @Test
    fun ignoresCommentsBlankLinesAndTitle() {
        val text = "# made by hand\n\nTITLE \"x\"\n" + identity2.lines().drop(1).joinToString("\n")
        assertEquals(2, parseCube(text).size)
    }

    @Test
    fun rejectsWrongEntryCount() {
        val short = identity2.lines().dropLast(1).joinToString("\n")
        val e = assertFailsWith<IllegalArgumentException> { parseCube(short) }
        assertEquals(true, e.message!!.contains("expected 8") && e.message!!.contains("got 7"))
    }

    @Test
    fun rejectsMissingSizeAnd1dTables() {
        assertFailsWith<IllegalArgumentException> { parseCube("0 0 0\n1 1 1") }
        assertFailsWith<IllegalArgumentException> { parseCube("LUT_1D_SIZE 2\n0 0 0\n1 1 1") }
    }

    @Test
    fun normalisesCustomDomain() {
        val text = "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n" + identity2.lines().drop(2).joinToString("\n") { l ->
            l.split(" ").joinToString(" ") { (it.toFloat() * 2f).toString() }
        }
        val lut = parseCube(text)
        assertEquals(listOf(1f, 1f, 1f), lut.rgb.slice(21..23))
    }

    @Test
    fun parsesTheShippedHyperRealCube() {
        val lut = parseCube(File("src/main/assets/hyperreal_rec709_33.cube").readText())
        assertEquals(33, lut.size)
        assertEquals(33 * 33 * 33 * 3, lut.rgb.size)
        assertEquals(0.000421f, lut.rgb[1], 1e-6f)
        assertEquals(0.998789f, lut.rgb[lut.rgb.size - 1], 1e-6f)
    }
}

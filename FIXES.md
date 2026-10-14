# GameLens — fixes applied in this rebuild

1. app/build.gradle.kts — ONNX Runtime QNN pinned to 1.20.0 (was 1.24.3,
   which does not resolve on Maven and kills the build at dependency resolution).

2. AiPass.kt — atlas internal format is RGBA32F (was RGB32F). Adreno
   drivers refuse RGB32F as color-renderable. The non-planar upload path
   expands model RGB output to RGBA before glTexSubImage2D.

3. NpuUpscaler.kt — Os.setenv wrapped in runCatching; it can throw
   ErrnoException on some ROMs and env is best-effort anyway.

4. AiSettings.kt — viewMode default is 0 (AI only). Was 1 (split).

5. MainActivity.kt — build label bumped to 0.4 to match versionName.

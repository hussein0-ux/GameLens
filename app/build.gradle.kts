plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.gamelens"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.gamelens"
        minSdk = 33
        targetSdk = 34
        versionCode = 4
        versionName = "0.4"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    packaging {
        // QNN loads its HTP libraries by name from the app's native library directory,
        // so they must be extracted to disk instead of read from inside the APK.
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += setOf("**/libc++_shared.so")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.1")
    // Same LiteRT versions as NeuralUpscalerApp. Its Kotlin API needs Kotlin 2.x (see the root build.gradle.kts).
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    implementation("com.google.ai.edge.litert:litert-gpu:2.3.0")
    implementation("com.google.ai.edge.litert:litert-npu-runtime-qualcomm:2.3.0")
    testImplementation("org.jetbrains.kotlin:kotlin-test:1.9.23")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:1.9.23")
}

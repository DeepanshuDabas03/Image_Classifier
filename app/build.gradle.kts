import java.net.URL

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.image.classifier"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.image.classifier"
        minSdk = 30
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        viewBinding = true
        compose=true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.10"
    }
    // Download  "mobilenetv1.tflite" into the "assets" directory.


    task("downloadModel") {
        doLast {
            val url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/android/mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite"
            val file = File(projectDir, "src/main/assets/mobilenetv1.tflite")
            if (!file.exists()) {
                file.parentFile.mkdirs()
                file.createNewFile()
                file.writeBytes(URL(url).readBytes())
            }
        }
    }
    tasks.getByName("preBuild").dependsOn("downloadModel")
    ndkVersion = "25.1.8937393"
    buildToolsVersion = "34.0.0"


}

dependencies {

    implementation( "androidx.compose.ui:ui:1.6.6")
    implementation( "androidx.compose.material:material:1.6.6")
    implementation ("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    implementation ("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-metadata:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
}




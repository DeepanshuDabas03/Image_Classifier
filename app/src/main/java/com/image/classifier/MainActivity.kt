package com.image.classifier

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.image.classifier.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContent {
            ClassifierApp()
        }
    }


}
class NativeBundle {
    companion object {
        init {
            System.loadLibrary("classifier")
        }
    }

    // Declare the native function.
    external fun stringFromJNI(): String
    external fun preprocessImage(image: Bitmap): ByteBuffer
}

@Composable
fun ClassifierApp() {
    var image by remember { mutableStateOf<ImageBitmap?>(null) }
    var output by remember { mutableStateOf("") }
    val context = LocalContext.current
    val getContent = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val inputStream = context.contentResolver.openInputStream(it)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            image = bitmap.asImageBitmap()
        }
    }
    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Open the gallery to select an image or take a photo
        Button(onClick = {
            getContent.launch("image/*")

        }) {
            Text(text = "Select Image")
        }


        Button(onClick = {

            image?.let {
                val classifier = ImageClassifier(context)

                val result = classifier.classifyImage(it)
                output = "Classified as $result"
            }
        }) {
            Text("Classify Image")
        }

        image?.let {
            Image(
                bitmap = it,
                contentDescription = "Loaded image",
                modifier = Modifier.size(200.dp)
            )
        }
        Spacer(modifier = Modifier.size(50.dp))
        Text(output,color=androidx.compose.ui.graphics.Color.Magenta,style=androidx.compose.ui.text.TextStyle(fontSize = 20.sp))
    }
}

class ImageClassifier(private val context: Context) {
    val tfLite: Interpreter
    val options = Interpreter.Options()
    var nnApiDelegate: NnApiDelegate? = null
    val assetManager = context.assets

    init {
        nnApiDelegate = NnApiDelegate()
        options.addDelegate(nnApiDelegate)

        try {
            tfLite = Interpreter(loadModelFile(assetManager, "mobilenetv1.tflite"), options)

        } catch (e: Exception) {
            throw RuntimeException(e)
        }
    }


    private fun loadModelFile(assetManager: AssetManager, modelFilename: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    fun classifyImage(image: ImageBitmap): String {

        val input = preprocessImage(image)
        val output = Array(1) { ByteArray(1001) }

        tfLite.run(input, output)
        val classScores = output[0].map { it.toUByte().toInt() }.toIntArray()

        val maxIndex = classScores.indices.maxByOrNull { classScores[it] } ?: -1

        return "Class ${classScores[maxIndex] } ."
    }

    private fun preprocessImage(image: ImageBitmap): ByteBuffer {
        // Resize the image to the input size of the model
        val imgSize = 224
        val native=NativeBundle()
        val result=native.preprocessImage(image.asAndroidBitmap())
        val resizedImage = Bitmap.createScaledBitmap(image.asAndroidBitmap(), imgSize, imgSize, true)

        // Normalize the pixels
        val byteBuffer = ByteBuffer.allocateDirect(  imgSize * imgSize * 3 )

        byteBuffer.order(ByteOrder.nativeOrder())
        for (y in 0 until imgSize) {
            for (x in 0 until imgSize) {
                val pixel = resizedImage.getPixel(x, y)
                byteBuffer.put(Color.red(pixel).toByte())
                byteBuffer.put(Color.green(pixel).toByte())
                byteBuffer.put(Color.blue(pixel).toByte())
            }
        }
//        return result
        return byteBuffer

    }

    protected fun finalize() {
        try {
            nnApiDelegate?.close()
            tfLite.close()
        }
        catch (e: Throwable) {
            throw e
        }
    }

}




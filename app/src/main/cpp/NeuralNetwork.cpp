#include <jni.h>
#include <android/bitmap.h>
#include <android/sharedmem.h>
#include <sys/mman.h>
#include <unistd.h>
#include <android/NeuralNetworks.h>
#include <string.h>
#include <vector>

jobject createBitmapFromOutputData(JNIEnv* env, float* outputData, jlong* outputDimensions) {
    // Get the Bitmap class
    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapMethod = env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");

    // Get the Bitmap Config class and the ARGB_8888 field
    jclass bitmapConfigClass = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888FieldID = env->GetStaticFieldID(bitmapConfigClass, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    jobject argb8888 = env->GetStaticObjectField(bitmapConfigClass, argb8888FieldID);

    // Create an empty bitmap with the given dimensions and config
    jobject bitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethod,224, 224, argb8888);

    // Get the setPixel method ID
    jmethodID setPixelMethod = env->GetMethodID(bitmapClass, "setPixel", "(III)V");

    // Set the pixels
    for (jlong y = 0; y < 224; ++y) {
        for (jlong x = 0; x < 224; ++x) {
            // Convert the output data to an integer color value (ARGB)
            int color =static_cast<jlong> (outputData[y * outputDimensions[0] + x] * 255.0f) & 0xFF;
            color = color | (color << 8) | (color << 16) | (0xFF << 24);

            // Set the pixel color in the bitmap
            env->CallVoidMethod(bitmap, setPixelMethod, x, y, color);
        }
    }

    return bitmap;
}




extern "C" JNIEXPORT jobject JNICALL
Java_com_image_classifier_NativeBundle_preprocessImage(
        JNIEnv* env, jobject /* this */, jobject bitmap) {
    // Get the bitmap info and pixels.
    AndroidBitmapInfo bitmapInfo;
    AndroidBitmap_getInfo(env, bitmap, &bitmapInfo);
    void* bitmapPixels;
    AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels);

    // Create a shared memory object from the bitmap pixels.
    int fd = ASharedMemory_create("bitmapPixels", bitmapInfo.width * bitmapInfo.height * 4);
    void* mappedMemory = mmap(nullptr, bitmapInfo.width * bitmapInfo.height * 4, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    memcpy(mappedMemory, bitmapPixels, bitmapInfo.width * bitmapInfo.height * 4);
    ANeuralNetworksMemory* memory = nullptr;
    ANeuralNetworksMemory_createFromFd(bitmapInfo.width * bitmapInfo.height * 4, PROT_READ | PROT_WRITE, fd, 0, &memory);

    // Initialize the NNAPI.
    uint32_t inputDimensions[] = {1, bitmapInfo.height, bitmapInfo.width, 3};
    jlong outputDimensions[] = {(1),
                                     (224),
(224),
                                     (3)};

    ANeuralNetworksModel* model = nullptr;
    ANeuralNetworksModel_create(&model);

    // Define the operand types.
    ANeuralNetworksOperandType inputType, outputType;
    inputType.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    inputType.dimensionCount = 4; // NHWC format
    inputType.dimensions = inputDimensions;
    outputType.type = ANEURALNETWORKS_TENSOR_FLOAT32;
    outputType.dimensionCount = 4; // NHWC format
    outputType.dimensions = reinterpret_cast<const uint32_t *>(outputDimensions);

    // Add the operands to the model.
    ANeuralNetworksModel_addOperand(model, &inputType);
    ANeuralNetworksModel_addOperand(model, &outputType);

    // Define the operation.
    uint32_t inputIndexes[1] = {0};
    uint32_t outputIndexes[1] = {1};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_RESIZE_BILINEAR, 1, inputIndexes, 1, outputIndexes);

    // Finish the model.
    ANeuralNetworksModel_finish(model);

    // Compile the model.
    ANeuralNetworksCompilation* compilation = nullptr;
    ANeuralNetworksCompilation_create(model, &compilation);
    ANeuralNetworksCompilation_finish(compilation);

    // Set the input tensor.
    ANeuralNetworksExecution* execution = nullptr;
    ANeuralNetworksExecution_create(compilation, &execution);
    ANeuralNetworksExecution_setInputFromMemory(execution, 0, &inputType, memory, 0, bitmapInfo.width * bitmapInfo.height * 4);


    int outputFd = ASharedMemory_create("outputTensor", outputDimensions[0] * outputDimensions[1] * outputDimensions[2] * outputDimensions[3] * sizeof(float));
    void* outputMappedMemory = mmap(nullptr, outputDimensions[0] * outputDimensions[1] * outputDimensions[2] * outputDimensions[3] * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED, outputFd, 0);
    ANeuralNetworksMemory* outputMemory = nullptr;
    ANeuralNetworksMemory_createFromFd(outputDimensions[0] * outputDimensions[1] * outputDimensions[2] * outputDimensions[3] * sizeof(float), PROT_READ | PROT_WRITE, outputFd, 0, &outputMemory);
    ANeuralNetworksExecution_setOutputFromMemory(execution, 0, &outputType, outputMemory, 0, outputDimensions[0] * outputDimensions[1] * outputDimensions[2] * outputDimensions[3] * sizeof(float));
    // Run the execution.
    ANeuralNetworksExecution_startCompute(execution, nullptr);

    // Get the output tensor.
    std::vector<float> output(outputDimensions[0] * outputDimensions[1] * outputDimensions[2] * outputDimensions[3]);
    memcpy(output.data(), outputMappedMemory, outputDimensions[0] * outputDimensions[1] * outputDimensions[2] * outputDimensions[3] * sizeof(float));

    // Clean up.
    munmap(outputMappedMemory, outputDimensions[0] * outputDimensions[1] * outputDimensions[2] * outputDimensions[3] * sizeof(float));
    close(outputFd);
    ANeuralNetworksExecution_free(execution);
    ANeuralNetworksMemory_free(memory);
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);

    // Convert the output data back into a bitmap and return it.
    // This will depend on how you want to use the output data.
    jobject outputBitmap = createBitmapFromOutputData(env, output.data(), outputDimensions);
    return nullptr;
}

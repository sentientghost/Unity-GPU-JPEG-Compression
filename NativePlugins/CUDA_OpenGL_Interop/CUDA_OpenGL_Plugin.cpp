#include "pch.h"
#include "CUDA_OpenGL_Plugin.h"


// UnitySetInterfaces global variables
static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static UnityGfxRenderer s_RendererType = kUnityGfxRendererNull;

// CUDA global variables
cudaDeviceProp cuda_props;
unsigned char* cuda_img_buffer = nullptr;
size_t cuda_img_buffer_size;

// nvJPEG global variables
nvjpegHandle_t nv_handle;
nvjpegEncoderState_t nv_enc_state;
nvjpegEncoderParams_t nv_enc_params;
nvjpegImage_t nv_image;
size_t nv_stream_size;

// OpenGL global variables
unsigned char* gl_tex_image = nullptr;

// Texture global variables
static GLuint g_TextureHandle = NULL;
static int g_TextureWidth = 0;
static int g_TextureHeight = 0;
static float g_TextureQuality = 75;

// Debug global variables
typedef void (*FuncPtr)(const char*);
FuncPtr Debug;

// Native Times global variables
typedef void (*FuncPtr)(const char*);
FuncPtr SendNativeTimes;

// General global variables
std::vector<unsigned char> jpeg_data;
std::string dataPath = "";
float nativeTimes[3] = { 0, 0, 0 };



// --------------------------------------------------------------------------
// UNITY DEFINED FUNCTIONS

// Unity plugin load event
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
{
    s_UnityInterfaces = unityInterfaces;
    s_Graphics = unityInterfaces->Get<IUnityGraphics>();

    s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

    // Run OnGraphicsDeviceEvent(initialize) manually on plugin load
    // to not miss the event in case the graphics device is already initialized
    OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

// Unity plugin unload event
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

// GraphicsDeviceEvent
static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    switch (eventType)
    {
    case kUnityGfxDeviceEventInitialize:
    {
        s_RendererType = s_Graphics->GetRenderer();
        break;
    }
    case kUnityGfxDeviceEventShutdown:
    {
        s_RendererType = kUnityGfxRendererNull;
        break;
    }
    case kUnityGfxDeviceEventBeforeReset:
    {
        break;
    }
    case kUnityGfxDeviceEventAfterReset:
    {
        break;
    }
    };
}


// --------------------------------------------------------------------------
// EXPORTED FUNCTIONS TO C# SCRIPT 

// Function used for Debugging
extern "C" void UNITY_INTERFACE_EXPORT SetDebugFunction(FuncPtr fp)
{
    Debug = fp;
}

// Function used to send the native times metircs to C# script
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetNativeTimes(FuncPtr fp)
{
    SendNativeTimes = fp;
}

// Gets Unity Texture2D resource
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* textureHandle, int imageWidth, int imageHeight, int cameraQuality, char* path)
{
    g_TextureHandle = (GLuint)textureHandle;
    g_TextureWidth = imageWidth;
    g_TextureHeight = imageHeight;
    g_TextureQuality = cameraQuality;
    dataPath = path;

    // Set OpenGL texture image size where 3 is the number of channels (RGB)
    gl_tex_image = (unsigned char*)malloc(sizeof(unsigned char) * g_TextureWidth * g_TextureHeight * 3);

    // Set CUDA image buffer size 
    cuda_img_buffer_size = g_TextureWidth * g_TextureHeight * NVJPEG_MAX_COMPONENT;
}

// Get a rendering event callback function
extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
    return OnRenderEvent;
}


// --------------------------------------------------------------------------
// CUDA FUNCTIONS

// Function to copy image from OpenGl to CUDA
bool CopyImage()
{
    // Initialise error flag
    bool errFlag = false;

    // OpenGL bind texture and adapt the alignment requirements for the start of each pixel row
    glBindTexture(GL_TEXTURE_2D, g_TextureHandle);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, gl_tex_image);

    // Initialise CUDA device properties
    if (!errFlag)
        errFlag = CheckErrors(cudaGetDeviceProperties(&cuda_props, 0));

    // Initialise nvJPEG structures
    if (!errFlag)
        errFlag = CheckErrors(nvjpegCreateSimple(&nv_handle));

    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, nullptr));
 
    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, nullptr));

    // Set parameters
    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncoderParamsSetQuality(nv_enc_params, g_TextureQuality, nullptr));

    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 0, nullptr));

    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, nullptr));

    // Copy to CUDA buffer to allow GPU to have access to the memory
    if (!errFlag)
        errFlag = CheckErrors(cudaMalloc(reinterpret_cast<void**>(&cuda_img_buffer), cuda_img_buffer_size));

    if (!errFlag)
        errFlag = CheckErrors(cudaMemcpy(cuda_img_buffer, gl_tex_image, cuda_img_buffer_size, cudaMemcpyHostToDevice));

    // Fill nvjpegImage using planar RGB
    // 
    /*for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++)
    {
        nv_image.channel[i] = cuda_img_buffer + g_TextureWidth * g_TextureHeight * i;
        nv_image.pitch[i] = g_TextureWidth;
    }*/

    // Fill nvjpegImage using interleaved RGB
    nv_image.channel[0] = cuda_img_buffer;
    nv_image.pitch[0] = 3 * g_TextureWidth;

    return errFlag;
}

// Function to compress image
bool EncodeImage()
{
    // Initialise error flag
    bool errFlag = false;

    // Forces the program to wait for all previously issued commands in all streams on the device to finish before continuing
    cudaDeviceSynchronize();

    // Compress image
    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &nv_image, NVJPEG_INPUT_RGBI, g_TextureWidth, g_TextureHeight, nullptr));

    // Get compressed stream size
    if (!errFlag)
    {
        errFlag = CheckErrors(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &nv_stream_size, nullptr));
        jpeg_data.resize(nv_stream_size);
    }

    // Get stream itself
    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg_data.data(), &nv_stream_size, 0));

    return errFlag;
}

// Function to write jpegstream to specified file location
bool WriteImage()
{
    // Initialise error flag
    bool errFlag = false;

    const char* jpeg_stream = reinterpret_cast<const char*>(jpeg_data.data());
    std::ofstream outputFile(dataPath.c_str(), std::ios::out | std::ios::binary);

    if (outputFile)
    {
        outputFile.write(jpeg_stream, nv_stream_size);

        if (outputFile.bad())
        {
            Debug("Error Write Operation Failed");
            errFlag = true;
        }
    }
    else
    {
        Debug("Error Opening Output File");
        errFlag = true;
    }
           
    outputFile.close();

    return errFlag;
}

// Function to free resources
bool CleanUp()
{
    // Initialise error flag
    bool errFlag = false;

    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncoderParamsDestroy(nv_enc_params));

    if (!errFlag)
        errFlag = CheckErrors(nvjpegEncoderStateDestroy(nv_enc_state));

    if (!errFlag)
        errFlag = CheckErrors(nvjpegDestroy(nv_handle));

    return errFlag;
}


// --------------------------------------------------------------------------
// PLUG-IN SPECIFIC DEFINED FUNCTIONS

// Plugin function to handle a specific rendering event
static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
    // Initialise error flag
    bool errFlag = false;

    // COPY
    auto startTime = std::chrono::high_resolution_clock::now();
    if (!errFlag)
        CopyImage();
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = endTime - startTime;
    nativeTimes[0] = duration.count();

    // ENCODE
    startTime = std::chrono::high_resolution_clock::now();
    if (!errFlag)
        EncodeImage();
    endTime = std::chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    nativeTimes[1] = duration.count();
    
    // WRITE
    startTime = std::chrono::high_resolution_clock::now();
    if (!errFlag)
        WriteImage();
    endTime = std::chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    nativeTimes[2] = duration.count();

    // SEND METRICS
    std::string nativeTimesStr = std::to_string(nativeTimes[0]) + "," + std::to_string(nativeTimes[1]) + "," + std::to_string(nativeTimes[2]);
    SendNativeTimes(nativeTimesStr.c_str());

    // CLEAN UP
    CleanUp();

    if (errFlag)
        Debug("Error occured during execution");
}

// Overloaded function to print any nvJPEG errors
bool CheckErrors(nvjpegStatus_t err)
{
    bool errFlag;

    if (err == NVJPEG_STATUS_SUCCESS)
    {
        errFlag = false;
    }
    else
    {
        errFlag = true;

        // Find and output exact error 
        if (err == NVJPEG_STATUS_NOT_INITIALIZED)
        {
            Debug("NVJPEG_STATUS_NOT_INITIALIZED");
        }
        else if (err == NVJPEG_STATUS_INVALID_PARAMETER)
        {
            Debug("NVJPEG_STATUS_INVALID_PARAMETER");
        }
        else if (err == NVJPEG_STATUS_BAD_JPEG)
        {
            Debug("NVJPEG_STATUS_BAD_JPEG");
        }
        else if (err == NVJPEG_STATUS_JPEG_NOT_SUPPORTED)
        {
            Debug("NVJPEG_STATUS_JPEG_NOT_SUPPORTED");
        }
        else if (err == NVJPEG_STATUS_ALLOCATOR_FAILURE)
        {
            Debug("NVJPEG_STATUS_ALLOCATOR_FAILURE");
        }
        else if (err == NVJPEG_STATUS_EXECUTION_FAILED)
        {
            Debug("NVJPEG_STATUS_EXECUTION_FAILED");
        }
        else if (err == NVJPEG_STATUS_ARCH_MISMATCH)
        {
            Debug("NVJPEG_STATUS_ARCH_MISMATCH");
        }
        else if (err == NVJPEG_STATUS_INTERNAL_ERROR)
        {
            Debug("NVJPEG_STATUS_INTERNAL_ERROR");
        }
        else if (err == NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED)
        {
            Debug("NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED");
        }
        else
        {
            Debug("Unknown nvJpeg Error");
        }
    }

    return errFlag;
}

// Overloaded function to print any CUDA errors
bool CheckErrors(cudaError_t err)
{
    bool errFlag;

    if (err == cudaSuccess)
    {
        errFlag = false;
    }
    else
    {
        Debug(cudaGetErrorString(err));
        errFlag = true;
    }

    return errFlag;
}

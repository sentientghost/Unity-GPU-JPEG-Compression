#include "pch.h"
#include "CUDAPlugin.h"
#include "StructsAndEnums.h"

//typedef struct __device_builtin__ __align__(4) uchar4
//{
//    unsigned char x, y, z, w;
//} uchar4;

//texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> texRefUChar;

// UnitySetInterfaces global variables
static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static UnityGfxRenderer s_RendererType = kUnityGfxRendererNull;

// CUDA global variables
cudaGraphicsResource_t m_cudaGraphicsResource = NULL; /** Interop resource handle**/
cudaArray* m_cudaArray; /** CUDA array that the texture is mapped to **/
cudaTextureObject_t m_texture; /** reference to exture to read data through*/

// Texture global variables
static GLuint g_TextureHandle = NULL;
static int g_TextureWidth = 0;
static int g_TextureHeight = 0;

// Debug global variables
typedef void (*FuncPtr)(const char*);
FuncPtr Debug;
cudaError_t err;

struct encode_params_t {
    std::string input_dir;
    std::string output_dir;
    std::string format;
    std::string subsampling;
    int quality;
    int huf;
    int dev;
};

nvjpegHandle_t nv_handle;
nvjpegEncoderState_t nv_enc_state;
nvjpegEncoderParams_t nv_enc_params;
nvjpegJpegState_t jpeg_state;
cudaStream_t stream;

unsigned char* pixels = (unsigned char*)malloc(sizeof(unsigned char) * 256 * 256 * 3);



// --------------------------------------------------------------------------
// UNITY DEFINED FUNCTIONS

// Unity plugin load event
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
{
    s_UnityInterfaces = unityInterfaces;
    s_Graphics = unityInterfaces->Get<IUnityGraphics>();

    /*IUnityGraphicsD3D11* d3d = s_UnityInterfaces->Get<IUnityGraphicsD3D11>();
    globals::device = d3d->GetDevice();
    globals::device->GetImmediateContext(&globals::context);*/

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

extern "C" cudaError_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* textureHandle, int imageWidth, int imageHeight)
{
    // A script calls this at initialization time; just remember the texture pointer here.
    // Will update texture pixels each frame from the plugin rendering event (texture update
    // needs to happen on the rendering thread).
    g_TextureHandle = (GLuint)textureHandle;
    g_TextureWidth = imageWidth;
    g_TextureHeight = imageHeight;

    //cudaError_t err = cudaSuccess;
    return err;
}

std::string checkErrors(nvjpegStatus_t err)
{
    if (err == NVJPEG_STATUS_NOT_INITIALIZED)
        return "NVJPEG_STATUS_NOT_INITIALIZED";
    else if (err == NVJPEG_STATUS_INVALID_PARAMETER)
        return "NVJPEG_STATUS_INVALID_PARAMETER";
    else if (err == NVJPEG_STATUS_BAD_JPEG)
        return "NVJPEG_STATUS_BAD_JPEG";
    else if (err == NVJPEG_STATUS_JPEG_NOT_SUPPORTED)
        return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
    else if (err == NVJPEG_STATUS_ALLOCATOR_FAILURE)
        return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
    else if (err == NVJPEG_STATUS_EXECUTION_FAILED)
        return "NVJPEG_STATUS_EXECUTION_FAILED";
    else if (err == NVJPEG_STATUS_ARCH_MISMATCH)
        return "NVJPEG_STATUS_ARCH_MISMATCH";
    else if (err == NVJPEG_STATUS_INTERNAL_ERROR)
        return "NVJPEG_STATUS_INTERNAL_ERROR";
    else if (err == NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED)
        return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";

    return "Fault in System";
}

// Function used for Debugging
extern "C" void UNITY_INTERFACE_EXPORT SetDebugFunction(FuncPtr fp)
{
    Debug = fp;
}

//int dev_malloc(void** p, size_t s) { return (int)cudaMalloc(p, s); }
//int dev_free(void* p) { return (int)cudaFree(p); }

//template<typename T>
//inline __device__ T tex2D(cudaTextureObject_t texObject, float x, float y)
//{
//    T ret;
//    unsigned ret1, ret2, ret3, ret4;
//    __asm__("tex.2d.v4.u32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];" :
//    "=r"(ret1), "=r"(ret2), "=r"(ret3), "=r"(ret4) :
//        "l"(texObject), "f"(x), "f"(y));
//    conv(&ret, ret1, ret2, ret3, ret4);
//    return ret;
//}

int encodeImage(std::string sImagePath, std::string sOutputPath, double& time, nvjpegOutputFormat_t output_format, nvjpegInputFormat_t input_format)
{
    time = 0.;
    unsigned char* pBuffer = NULL;
    cudaEvent_t startEvent = NULL;
    cudaEvent_t stopEvent = NULL;
    float loopTime = 0;
    nvjpegStatus_t nvjpegErr;
    cudaError_t cudaErr;
    checkCudaErrors(cudaEventCreate(&startEvent, cudaEventBlockingSync));
    checkCudaErrors(cudaEventCreate(&stopEvent, cudaEventBlockingSync));
    
    // Allocates size bytes of linear memory on the device and returns in *devPtr a pointer to the allocated memory
    //cudaErr = cudaMalloc(&pixels, g_TextureWidth * g_TextureHeight * NVJPEG_MAX_COMPONENT);
    cudaErr = cudaMalloc(&pixels, g_TextureWidth * g_TextureHeight * 3);
    if (cudaErr == cudaSuccess)
        Debug("Success Encode 1");
    else
        Debug(cudaGetErrorString(cudaErr));

    //uchar4 temp = tex2D<uchar4>(m_texture, g_TextureWidth, g_TextureHeight);

    /*nvjpegImage_t imgdesc =
    {
        {
            pixels,
            pixels + g_TextureWidth * g_TextureHeight,
            pixels + g_TextureWidth * g_TextureHeight * 2,
            pixels + g_TextureWidth * g_TextureHeight * 3
        },
        {
            (unsigned int)g_TextureWidth,
            (unsigned int)g_TextureWidth,
            (unsigned int)g_TextureWidth,
            (unsigned int)g_TextureWidth
        }
    };*/

    /*nvjpegImage_t imgdesc;
    for (int i = 0; i < 3; i++) {
        cudaMalloc((void**)&(imgdesc.channel[i]), g_TextureWidth * g_TextureHeight);
        cudaMemcpy(imgdesc.channel[i], &pixels[i], g_TextureWidth * g_TextureHeight, cudaMemcpyHostToDevice);
        imgdesc.pitch[i] = (size_t)g_TextureWidth;
    }*/

    /*nvjpegImage_t imgdesc;
    for (int i = 0; i < 3; i++) {
        imgdesc.channel[i] = (size_t)(g_TextureWidth * g_TextureHeight);
        imgdesc.pitch[i] = (size_t)g_TextureWidth;
    }*/

    nvjpegImage_t imgdesc;
    //imgdesc.channel[0] = pixels;
    //imgdesc.pitch[0] = 3 * g_TextureWidth;

    stbi_write_jpg("F:/stbi.jpg", 256, 256, 3, pixels, 75);

    imgdesc.channel[0] = pixels;
    imgdesc.channel[1] = pixels + g_TextureWidth * g_TextureHeight;
    imgdesc.channel[2] = pixels + 2 * g_TextureWidth * g_TextureHeight;
    imgdesc.pitch[0] = g_TextureWidth;
    imgdesc.pitch[1] = g_TextureWidth;
    imgdesc.pitch[2] = g_TextureWidth;

    //cudaDeviceSynchronize();

    //nvjpeg();

    // cudaDeviceSynchronize();

    checkCudaErrors(cudaEventRecord(startEvent, NULL));

    // ENCODE
    nvjpegErr = nvjpegEncodeImage(nv_handle,
        nv_enc_state,
        nv_enc_params,
        &imgdesc,
        NVJPEG_INPUT_RGB,
        g_TextureWidth,
        g_TextureHeight,
        stream);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Encode 2");
    else
        Debug(checkErrors(nvjpegErr).c_str());
    
    //std::vector<unsigned char> obuffer;
    size_t length;
    nvjpegErr = nvjpegEncodeRetrieveBitstream(
        nv_handle,
        nv_enc_state,
        NULL,
        &length,
        stream);

    cudaStreamSynchronize(stream);

    //unsigned char* obuffer = (unsigned char*)malloc(length);
    std::vector<unsigned char> jpeg(length);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Encode 3");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    nvjpegErr = nvjpegEncodeRetrieveBitstream(
        nv_handle,
        nv_enc_state,
        jpeg.data(),
        &length,
        0);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Encode 4");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    checkCudaErrors(cudaEventRecord(stopEvent, NULL));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    double encoder_time = static_cast<double>(loopTime);

    //stbi_write_jpg("F:/stbiTest.jpg", 256, 256, 4, obuffer, 75);

    // write stream to file
    const char* jpegstream = reinterpret_cast<const char*>(jpeg.data());
    cudaStreamSynchronize(stream);
    std::string output_filename = "F:/Test1.jpg";
    std::ofstream outputFile(output_filename.c_str(), std::ios::out | std::ios::binary);
    //outputFile.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<int>(length));
    outputFile.write(jpegstream, length);
    outputFile.close();

    // Free memory
    checkCudaErrors(cudaFree(pixels));

    //time = encoder_time;

    return 0;
}

int processImage(encode_params_t param)
{
    std::string sInputPath(param.input_dir);
    std::string sOutputPath(param.output_dir);
    std::string sFormat(param.format);
    std::string sSubsampling(param.subsampling);
    nvjpegOutputFormat_t oformat = NVJPEG_OUTPUT_RGB;
    nvjpegInputFormat_t iformat = NVJPEG_INPUT_RGB;
    nvjpegStatus_t nvjpegErr;

    int error_code = 1;

    nvjpegErr = nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Process 1");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    //checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_420, NULL));

    double total_time = 0.;
    double encoder_time = 0.;

    int image_error_code = encodeImage("test.jpg", sOutputPath, encoder_time, oformat, iformat);

    std::stringstream ss;
    ss << "Encode Time: " << encoder_time;
    std::string s = ss.str();
    Debug(s.c_str());

    return 0;
}

void SaveImage()
{
    // Inputs
    std::string sOutputPath;
    nvjpegOutputFormat_t output_format;
    nvjpegInputFormat_t input_format;
    cudaError_t cudaErr;
    nvjpegStatus_t nvjpegErr;

    //// begin function
    //double time = 0.;
    //cudaEvent_t startEvent = NULL, stopEvent = NULL;
    //float loopTime = 0;
    //checkCudaErrors(cudaEventCreate(&startEvent, cudaEventBlockingSync));
    //checkCudaErrors(cudaEventCreate(&stopEvent, cudaEventBlockingSync));

    //// initialize nvjpeg structures
    //nvjpegCreateSimple(&nv_handle);
    //nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream);
    //nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream);

    //nvjpegImage_t nv_image;

    // PARAMS
    //encode_params_t params = { "./test.jpg", "./test1.jpg", "yuv", "420", 75, 0, 0 };
    encode_params_t params = {"./test.jpg", "F:/test1.jpg", "rgb", "420", 75, 0, 0};

    // CUDA DEVICE PROPERTIES
    cudaDeviceProp props;
    cudaErr = cudaGetDeviceProperties(&props, params.dev);
    if (cudaErr == cudaSuccess)
        Debug("Success Save 1");
    else
        Debug(cudaGetErrorString(cudaErr));

    std::stringstream ss;
    ss << "Using GPU " << params.dev << " (" << props.name << ", " << props.multiProcessorCount << " SMs, " << props.maxThreadsPerMultiProcessor << " th/SM max, CC " << props.major << "." << props.minor << ", ECC " << props.ECCEnabled << ")";
    std::string s = ss.str();
    Debug(s.c_str());

    //nvjpegDevAllocator_t dev_allocator = { &dev_malloc, &dev_free };

    //consider other backends for nvjpeg
    nvjpegErr = nvjpegCreateSimple(&nv_handle);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 2");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    /*nvjpegErr = (nvjpegJpegStateCreate(nv_handle, &jpeg_state));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 3");
    else
        Debug(checkErrors(nvjpegErr).c_str());*/

    nvjpegErr = (nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 4");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    nvjpegErr = (nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 5");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    // sample input parameters
    nvjpegErr = (nvjpegEncoderParamsSetQuality(nv_enc_params, params.quality, stream));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 6");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    nvjpegErr = (nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, params.huf, stream));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 7");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    // process id
    int pidx = processImage(params);

    // CLEAN UP
    nvjpegErr = (nvjpegEncoderParamsDestroy(nv_enc_params));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 8");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    nvjpegErr = (nvjpegEncoderStateDestroy(nv_enc_state));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 9");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    /*nvjpegErr = (nvjpegJpegStateDestroy(jpeg_state));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 10");
    else
        Debug(checkErrors(nvjpegErr).c_str());*/

    nvjpegErr = (nvjpegDestroy(nv_handle));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 11");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    ////consider other backends for nvjpeg
    //checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nv_handle));
    //checkCudaErrors(nvjpegJpegStateCreate(nv_handle, &jpeg_state));
    //checkCudaErrors(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, NULL));
    //checkCudaErrors(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, NULL));

    //// sample input parameters
    //checkCudaErrors(nvjpegEncoderParamsSetQuality(nv_enc_params, params.quality, NULL));
    //checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, params.huf, NULL));

    //// process id
    //int pidx = processImage(params);

    //// CLEAN UP
    //checkCudaErrors(nvjpegEncoderParamsDestroy(nv_enc_params));
    //checkCudaErrors(nvjpegEncoderStateDestroy(nv_enc_state));
    //checkCudaErrors(nvjpegJpegStateDestroy(jpeg_state));
    //checkCudaErrors(nvjpegDestroy(nv_handle));
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
    //glGenTextures(1, &g_TextureHandle);
    //use 4 cause its RGBA
    std::vector<GLubyte> temptest(g_TextureWidth * g_TextureHeight * 4);

    unsigned char* texture_bytes = (unsigned char*)malloc(sizeof(unsigned char) * 256 * 256 * 4);


    glBindTexture(GL_TEXTURE_2D, g_TextureHandle);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    //glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, temptest.data());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    //err = cudaGraphicsGLRegisterImage(&m_cudaGraphicsResource, g_TextureHandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);


    //stbi_write_jpg("F:/stbi.jpg", 256, 256, 3, pixels, 75);

    //////////////////////////////////////////////
    // Save Image
    cudaError_t cudaErr;
    nvjpegStatus_t nvjpegErr;

    // PARAMS
    //encode_params_t params = { "./test.jpg", "./test1.jpg", "yuv", "420", 75, 0, 0 };
    encode_params_t params = { "./test.jpg", "F:/test1.jpg", "rgb", "420", 75, 0, 0 };

    // CUDA DEVICE PROPERTIES
    cudaDeviceProp props;
    cudaErr = cudaGetDeviceProperties(&props, params.dev);
    if (cudaErr == cudaSuccess)
        Debug("Success Save 1");
    else
        Debug(cudaGetErrorString(cudaErr));

    std::stringstream ss;
    ss << "Using GPU " << params.dev << " (" << props.name << ", " << props.multiProcessorCount << " SMs, " << props.maxThreadsPerMultiProcessor << " th/SM max, CC " << props.major << "." << props.minor << ", ECC " << props.ECCEnabled << ")";
    std::string s = ss.str();
    Debug(s.c_str());

    //nvjpegDevAllocator_t dev_allocator = { &dev_malloc, &dev_free };

    //consider other backends for nvjpeg
    nvjpegErr = nvjpegCreateSimple(&nv_handle);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 2");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    /*nvjpegErr = (nvjpegJpegStateCreate(nv_handle, &jpeg_state));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 3");
    else
        Debug(checkErrors(nvjpegErr).c_str());*/

    nvjpegErr = (nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, nullptr));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 4");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    nvjpegErr = (nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, nullptr));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 5");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    // sample input parameters
    nvjpegErr = (nvjpegEncoderParamsSetQuality(nv_enc_params, params.quality, nullptr));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 6");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    nvjpegErr = (nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 0, nullptr));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 7");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    ////////////////////////////////////////////
    // Process Image

    nvjpegErr = nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, nullptr);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Process 1");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    ///////////////////////////////////////////
    // Encode Image
    
    /*cudaErr = cudaMalloc(&pixels, g_TextureWidth * g_TextureHeight * 3);
    if (cudaErr == cudaSuccess)
        Debug("Success Encode 1");
    else
        Debug(cudaGetErrorString(cudaErr));*/

    //uchar4 temp = tex2D<uchar4>(m_texture, g_TextureWidth, g_TextureHeight);

    /*nvjpegImage_t imgdesc =
    {
        {
            pixels,
            pixels + g_TextureWidth * g_TextureHeight,
            pixels + g_TextureWidth * g_TextureHeight * 2,
            pixels + g_TextureWidth * g_TextureHeight * 3
        },
        {
            (unsigned int)g_TextureWidth,
            (unsigned int)g_TextureWidth,
            (unsigned int)g_TextureWidth,
            (unsigned int)g_TextureWidth
        }
    };*/

    /*nvjpegImage_t imgdesc;
    for (int i = 0; i < 3; i++) {
        cudaMalloc((void**)&(imgdesc.channel[i]), g_TextureWidth * g_TextureHeight);
        cudaMemcpy(imgdesc.channel[i], &pixels[i], g_TextureWidth * g_TextureHeight, cudaMemcpyHostToDevice);
        imgdesc.pitch[i] = (size_t)g_TextureWidth;
    }*/

    /*nvjpegImage_t imgdesc;
    for (int i = 0; i < 3; i++) {
        imgdesc.channel[i] = (size_t)(g_TextureWidth * g_TextureHeight);
        imgdesc.pitch[i] = (size_t)g_TextureWidth;
    }*/

    //nvjpegImage_t imgdesc;
    //imgdesc.channel[0] = pixels;
    //imgdesc.pitch[0] = 3 * g_TextureWidth;

    //stbi_write_jpg("F:/stbi.jpg", 256, 256, 3, pixels, 75);

  /*  imgdesc.channel[0] = pixels;
    imgdesc.channel[1] = pixels + g_TextureWidth * g_TextureHeight;
    imgdesc.channel[2] = pixels + 2 * g_TextureWidth * g_TextureHeight;
    imgdesc.pitch[0] = g_TextureWidth;
    imgdesc.pitch[1] = g_TextureWidth;
    imgdesc.pitch[2] = g_TextureWidth;*/

    //cudaDeviceSynchronize();

    //nvjpeg();

    // cudaDeviceSynchronize();

    //checkCudaErrors(cudaEventRecord(startEvent, NULL));

    unsigned char* img_buffer = nullptr;
    size_t img_buffer_size = g_TextureWidth * g_TextureHeight * NVJPEG_MAX_COMPONENT;

    cudaErr = cudaMalloc(reinterpret_cast<void**>(&img_buffer), img_buffer_size);
    if (cudaErr == cudaSuccess)
        Debug("Success Encode 1");
    else
        Debug(cudaGetErrorString(cudaErr));
    
    cudaErr = cudaMemcpy(img_buffer, pixels, img_buffer_size, cudaMemcpyHostToDevice);
    if (cudaErr == cudaSuccess)
        Debug("Success Encode 2");
    else
        Debug(cudaGetErrorString(cudaErr));



    nvjpegImage_t imgdesc;

    for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++)
        imgdesc.channel[i] = img_buffer + 256 * 256 * i;
    for (int i = 0; i < NVJPEG_MAX_COMPONENT; i++) imgdesc.pitch[i] = 256;


    cudaDeviceSynchronize();

    // ENCODE
    nvjpegErr = nvjpegEncodeImage(nv_handle,
        nv_enc_state,
        nv_enc_params,
        &imgdesc,
        NVJPEG_INPUT_RGB,
        g_TextureWidth,
        g_TextureHeight,
        nullptr);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Encode 2.5");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    //std::vector<unsigned char> obuffer;
    size_t length;
    nvjpegErr = nvjpegEncodeRetrieveBitstream(
        nv_handle,
        nv_enc_state,
        NULL,
        &length,
        stream);

    cudaStreamSynchronize(stream);

    //unsigned char* obuffer = (unsigned char*)malloc(length);
    std::vector<unsigned char> jpeg(length);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Encode 3");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    nvjpegErr = nvjpegEncodeRetrieveBitstream(
        nv_handle,
        nv_enc_state,
        jpeg.data(),
        &length,
        0);
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Encode 4");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    /*checkCudaErrors(cudaEventRecord(stopEvent, NULL));
    checkCudaErrors(cudaEventSynchronize(stopEvent));
    checkCudaErrors(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
    double encoder_time = static_cast<double>(loopTime);*/

    //stbi_write_jpg("F:/stbiTest.jpg", 256, 256, 4, obuffer, 75);

    // write stream to file
    const char* jpegstream = reinterpret_cast<const char*>(jpeg.data());
    cudaStreamSynchronize(stream);
    std::string output_filename = "F:/Test1.jpg";
    std::ofstream outputFile(output_filename.c_str(), std::ios::out | std::ios::binary);
    //outputFile.write(reinterpret_cast<const char*>(jpeg.data()), static_cast<int>(length));
    outputFile.write(jpegstream, length);
    outputFile.close();

    // Free memory
    //checkCudaErrors(cudaFree(pixels));

    ///////////////////////////////////////////
    
    // CLEAN UP
    nvjpegErr = (nvjpegEncoderParamsDestroy(nv_enc_params));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 8");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    nvjpegErr = (nvjpegEncoderStateDestroy(nv_enc_state));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 9");
    else
        Debug(checkErrors(nvjpegErr).c_str());

    /*nvjpegErr = (nvjpegJpegStateDestroy(jpeg_state));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 10");
    else
        Debug(checkErrors(nvjpegErr).c_str());*/

    nvjpegErr = (nvjpegDestroy(nv_handle));
    if (nvjpegErr == NVJPEG_STATUS_SUCCESS)
        Debug("Success Save 11");
    else
        Debug(checkErrors(nvjpegErr).c_str()); 

    //////////////////////////////////////////////

    //err = cudaGraphicsGLRegisterImage(&m_cudaGraphicsResource, g_TextureHandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);

    /*cudaGraphicsMapResources(1, &m_cudaGraphicsResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&m_cudaArray, m_cudaGraphicsResource, 0, 0);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_cudaArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr);*/

    //Debug(cudaGetErrorString(err));
    
    //SaveImage();

    /*cudaDestroyTextureObject(m_texture);
    cudaFreeArray(m_cudaArray);*/
}


// --------------------------------------------------------------------------
// GetRenderEventFunc, an example function we export which is used to get a rendering event callback function.

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
    return OnRenderEvent;
}

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetErrorString(cudaError_t error) {
    Debug(cudaGetErrorString(error));
}

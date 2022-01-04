#include "pch.h"
#include "CUDAPlugin.h"


// UnitySetInterfaces global variables
static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static UnityGfxRenderer s_RendererType = kUnityGfxRendererNull;

// CUDA global variables
cudaGraphicsResource_t resource = NULL;

// Texture global variables
static GLuint g_TextureHandle = NULL;
static int g_TextureWidth = 0;
static int g_TextureHeight = 0;

// Debug global variables
typedef void (*FuncPtr)(const char*);
FuncPtr Debug;
cudaError_t err;


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

// Function used for Debugging
extern "C" void UNITY_INTERFACE_EXPORT SetDebugFunction(FuncPtr fp)
{
    Debug = fp;
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
    //Debug("Success");
    err = cudaGraphicsGLRegisterImage(&resource, g_TextureHandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
    Debug(cudaGetErrorString(err));
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

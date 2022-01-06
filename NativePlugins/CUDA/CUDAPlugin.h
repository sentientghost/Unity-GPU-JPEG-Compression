#pragma once

#define DLL_EXPORT __declspec(dllexport)

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h> 
#include "nvjpeg.h"
#include "helper_nvJPEG.hxx"

#include <GL/glut.h>
#include "stb_image_write.h"


//#include "GL\glew.h"
//#include "GL\freeglut.h"
//#include "GL\glext.h"


//#include <memory>
//#include <x86intrin.h>

//#include <cuda_texture_types.h>
//
//#include "GL/glut.h"
//#include "GL/glext.h"
//
//#include <device_launch_parameters.h>
//#include <curand_kernel.h>
//#include <cuda.h>
//#include "texture_types.h"
//#include "cuda_texture_types.h"
//#include <texture_fetch_functions.h>
//#include <texture_indirect_functions.h>
//#include <helper_cuda.h>

// --------------------------------------------------------------------------
// UNITY DEFINED FUNCTIONS

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload();

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

// --------------------------------------------------------------------------
// EXPORTED FUNCTIONS TO C# SCRIPT 

extern "C" cudaError_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* textureHandle, int imageWidth, int imageHeight);

static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc();

extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API GetErrorString(cudaError_t error);

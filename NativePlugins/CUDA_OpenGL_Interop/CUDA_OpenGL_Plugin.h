#pragma once

#define DLL_EXPORT __declspec(dllexport)

#include "IUnityInterface.h"
#include "IUnityGraphics.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <GL/glut.h>

#include <helper_functions.h> 
#include "nvjpeg.h"
#include "helper_nvJPEG.hxx"


// --------------------------------------------------------------------------
// UNITY DEFINED FUNCTIONS

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload();

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

// --------------------------------------------------------------------------
// EXPORTED FUNCTIONS TO C# SCRIPT 

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetCopyTime();

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetEncodeTime();

extern "C" int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetWriteTime();

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* textureHandle, int imageWidth, int imageHeight, int cameraQuality, char* path);

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc();

// --------------------------------------------------------------------------
// CUDA FUNCTIONS

bool CopyImage();

bool EncodeImage();

bool WriteImage();

bool CleanUp();

// --------------------------------------------------------------------------
// PLUG-IN SPECIFIC DEFINED FUNCTIONS

static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

bool CheckErrors(nvjpegStatus_t err);

bool CheckErrors(cudaError_t err);

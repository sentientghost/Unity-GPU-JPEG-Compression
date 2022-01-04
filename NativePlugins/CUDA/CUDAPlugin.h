#pragma once

#define DLL_EXPORT __declspec(dllexport)

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


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

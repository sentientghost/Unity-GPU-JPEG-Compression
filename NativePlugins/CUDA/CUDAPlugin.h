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

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void * textureHandle, int imageWidth, int imageHeight);

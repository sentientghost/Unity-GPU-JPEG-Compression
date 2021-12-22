#pragma once

#define DLL_EXPORT __declspec(dllexport)

#include "IUnityGraphics.h"
#include "d3d11.h"
#include "IUnityGraphicsD3D11.h"
#include <ScreenGrab.h>
#include <wrl/client.h>
#include <wincodec.h>


//class D3D11API
//{
//public:
//	virtual ~D3D11API() { }
//
//	// Process general event like initialization, shutdown, device loss/reset etc.
//	virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces) = 0;
//
//};
//
//// Create a graphics API implementation instance for the given API type.
//D3D11API* CreateRenderAPI(UnityGfxRenderer apiType);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload();

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(ID3D11Resource* textureHandle, int w, int h);

static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc();
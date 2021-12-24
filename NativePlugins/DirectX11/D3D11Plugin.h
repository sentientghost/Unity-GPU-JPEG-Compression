#pragma once

#define DLL_EXPORT __declspec(dllexport)

#include "IUnityGraphics.h"
#include "d3d11.h"
#include "IUnityGraphicsD3D11.h"
#include "DirectXHelpers.h"
#include "../DirectXTK/Src/LoaderHelpers.h"
#include "../DirectXTK/Src/PlatformHelpers.h"


// Added requried namespaces
using Microsoft::WRL::ComPtr;


// --------------------------------------------------------------------------
// UNITY DEFINED FUNCTIONS

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload();

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

// --------------------------------------------------------------------------
// EXPORTED FUNCTIONS TO C# SCRIPT 

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API FillNativeTimes(float* data, int count);

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(ID3D11Resource * textureHandle, int imageWidth, int imageHeight, float cameraQuality, char* path);

// --------------------------------------------------------------------------
// DIRECTX TOOLKIT ADAPTED FUNCTIONS

HRESULT CaptureTexture(_In_ ID3D11DeviceContext* pContext, _In_ ID3D11Resource* pSource, D3D11_TEXTURE2D_DESC& desc, ComPtr<ID3D11Texture2D>& pStaging) noexcept;

HRESULT EncodeTexture(ID3D11DeviceContext* pContext, REFGUID guidContainerFormat, const wchar_t* filePath, const GUID* targetFormat, std::function<void(IPropertyBag2*)> setCustomProps, bool forceSRGB);

HRESULT SaveWICTextureToFile(const wchar_t* filePath);

// --------------------------------------------------------------------------
// PLUG-IN SPECIFIC DEFINED FUNCTIONS

std::string FormatError(unsigned __int32 hr);

std::wstring StringToWideString(const std::string& s);

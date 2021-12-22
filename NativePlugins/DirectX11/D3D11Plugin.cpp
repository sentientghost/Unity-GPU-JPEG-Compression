#include "pch.h"
#include "D3D11Plugin.h"
#include <Windows.h>
#include <wrl.h>
#include <d3d11_1.h>
#include <cassert>
#include "DirectXHelpers.h"

#include "../DirectXTK/Src/LoaderHelpers.h"
//#include "../DirectXTK/Src/PlatformHelpers.h"
//#include "../DirectXTK/Src/DDS.h"
//#include <DDSTextureLoader.h>

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace DirectX::LoaderHelpers;
using namespace Microsoft::WRL;

namespace globals {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
} // namespace globals


// --------------------------------------------------------------------------
// Debug Event

typedef void (*FuncPtr)(const char*);
FuncPtr Debug;

extern "C" void UNITY_INTERFACE_EXPORT SetDebugFunction(FuncPtr fp)
{
    Debug = fp;
}

// --------------------------------------------------------------------------
// UnitySetInterfaces

static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static UnityGfxRenderer s_RendererType = kUnityGfxRendererNull;

// Unity plugin load event
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
{
    s_UnityInterfaces = unityInterfaces;
    s_Graphics = unityInterfaces->Get<IUnityGraphics>();
    IUnityGraphicsD3D11* d3d = s_UnityInterfaces->Get<IUnityGraphicsD3D11>();
    globals::device = d3d->GetDevice();
    globals::device->GetImmediateContext(&globals::context);

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


// --------------------------------------------------------------------------
// GraphicsDeviceEvent

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    switch (eventType)
    {
        case kUnityGfxDeviceEventInitialize:
        {
            s_RendererType = s_Graphics->GetRenderer();
            //TODO: user initialization code
            break;
        }
        case kUnityGfxDeviceEventShutdown:
        {
            s_RendererType = kUnityGfxRendererNull;
            //TODO: user shutdown code
            break;
        }
        case kUnityGfxDeviceEventBeforeReset:
        {
            //TODO: user Direct3D 9 code
            break;
        }
        case kUnityGfxDeviceEventAfterReset:
        {
            //TODO: user Direct3D 9 code
            break;
        }
    };
}


// --------------------------------------------------------------------------
// SetTextureFromUnity, an example function we export which is called by one of the scripts.

static ID3D11Resource* g_TextureHandle = NULL;
static int   g_TextureWidth = 0;
static int   g_TextureHeight = 0;

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(ID3D11Resource* textureHandle, int w, int h)
{
	// A script calls this at initialization time; just remember the texture pointer here.
	// Will update texture pixels each frame from the plugin rendering event (texture update
	// needs to happen on the rendering thread).
	g_TextureHandle = textureHandle;
	g_TextureWidth = w;
	g_TextureHeight = h;
    Debug("Texture Received");
}


// --------------------------------------------------------------------------
// OnRenderEvent
// This will be called for GL.IssuePluginEvent script calls; eventID will
// be the integer passed to IssuePluginEvent. In this example, we just ignore
// that value.

std::string format_error(unsigned __int32 hr)
{
    std::stringstream ss;
    ss << "Error code = 0x" << std::hex << hr << std::endl;
    return ss.str();
}

HRESULT CaptureTexture(ID3D11DeviceContext* pContext, ID3D11Resource* pSource, D3D11_TEXTURE2D_DESC& desc, ComPtr<ID3D11Texture2D>& pStaging)
{
    if (!pContext || !pSource)
        return E_INVALIDARG;

    D3D11_RESOURCE_DIMENSION resType = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    pSource->GetType(&resType);

    if (resType != D3D11_RESOURCE_DIMENSION_TEXTURE2D)
    {
        Debug("ERROR: ScreenGrab does not support 1D or volume textures. Consider using DirectXTex instead.\n");
        return HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
    }

    ComPtr<ID3D11Texture2D> pTexture;
    HRESULT hr = pSource->QueryInterface(IID_GRAPHICS_PPV_ARGS(pTexture.GetAddressOf()));
    if (FAILED(hr))
        return hr;

    assert(pTexture);

    pTexture->GetDesc(&desc);

    if (desc.ArraySize > 1 || desc.MipLevels > 1)
    {
        Debug("WARNING: ScreenGrab does not support 2D arrays, cubemaps, or mipmaps; only the first surface is written. Consider using DirectXTex instead.\n");
    }

    ComPtr<ID3D11Device> d3dDevice;
    pContext->GetDevice(d3dDevice.GetAddressOf());

    if (desc.SampleDesc.Count > 1)
    {
        // MSAA content must be resolved before being copied to a staging texture
        Debug("MSAA content must be resolved before being copied to a staging texture");
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;

        ComPtr<ID3D11Texture2D> pTemp;
        hr = d3dDevice->CreateTexture2D(&desc, nullptr, pTemp.GetAddressOf());
        if (FAILED(hr))
            return hr;

        assert(pTemp);

        DXGI_FORMAT fmt = EnsureNotTypeless(desc.Format);

        UINT support = 0;
        hr = d3dDevice->CheckFormatSupport(fmt, &support);
        if (FAILED(hr))
            return hr;

        if (!(support & D3D11_FORMAT_SUPPORT_MULTISAMPLE_RESOLVE))
            return E_FAIL;

        for (UINT item = 0; item < desc.ArraySize; ++item)
        {
            for (UINT level = 0; level < desc.MipLevels; ++level)
            {
                UINT index = D3D11CalcSubresource(level, item, desc.MipLevels);
                pContext->ResolveSubresource(pTemp.Get(), index, pSource, index, fmt);
            }
        }

        desc.BindFlags = 0;
        desc.MiscFlags &= D3D11_RESOURCE_MISC_TEXTURECUBE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.Usage = D3D11_USAGE_STAGING;

        hr = d3dDevice->CreateTexture2D(&desc, nullptr, pStaging.ReleaseAndGetAddressOf());
        if (FAILED(hr))
            return hr;

        assert(pStaging);

        pContext->CopyResource(pStaging.Get(), pTemp.Get());
    }
    else if ((desc.Usage == D3D11_USAGE_STAGING) && (desc.CPUAccessFlags & D3D11_CPU_ACCESS_READ))
    {
        // Handle case where the source is already a staging texture we can use directly
        Debug("source is already a staging texture we can use directly");
        pStaging = pTexture;
    }
    else
    {
        // Otherwise, create a staging texture from the non-MSAA source
        Debug("create a staging texture from the non-MSAA source");
        desc.BindFlags = 0;
        desc.MiscFlags &= D3D11_RESOURCE_MISC_TEXTURECUBE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.Usage = D3D11_USAGE_STAGING;

        hr = d3dDevice->CreateTexture2D(&desc, nullptr, pStaging.ReleaseAndGetAddressOf());
        if (FAILED(hr))
            return hr;

        assert(pStaging);

        pContext->CopyResource(pStaging.Get(), pSource);
    }

    return S_OK;
}

static void SaveImage()
{
    std::string err;
    HRESULT hr = S_OK;

    /*ComPtr<ID3D11Texture2D> backBuffer;
    hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<LPVOID*>(backBuffer.GetAddressOf()));
    if (SUCCEEDED(hr))
    {
        hr = SaveWICTextureToFile(globals::context, backBuffer.Get(), GUID_ContainerFormatJpeg, L"F:/screenshots/SCREENSHOT.JPG");
    }
    DX::ThrowIfFailed(hr);*/

    hr = SaveWICTextureToFile(globals::context, g_TextureHandle, GUID_ContainerFormatJpeg, L"F:/screenshots/SCREENSHOT.JPG");
    

    //-----------------------------------------
    /*D3D11_TEXTURE2D_DESC desc = {};
    ComPtr<ID3D11Texture2D> pStaging;
    hr = CaptureTexture(globals::context, g_TextureHandle, desc, pStaging);

    if (FAILED(hr))
    {
        Debug("Capture Texture Failed");
    }
    else
    {
        Debug("Capture Texture Succeeded");
    }*/

    // Determine source format's WIC equivalent
    //WICPixelFormatGUID pfGuid = {};
    //bool sRGB = false;

    //ComPtr<ID3D11Texture2D> pTexture;
    //hr = g_TextureHandle->QueryInterface(IID_GRAPHICS_PPV_ARGS(pTexture.GetAddressOf()));
    //if (FAILED(hr))
    //    Debug("Query Inteface Failed");
    //else
    //    Debug("Query Interface Succeeded");

    //assert(pTexture);

    //pTexture->GetDesc(&desc);

    //if (desc.ArraySize > 1 || desc.MipLevels > 1)
    //{
    //    Debug("WARNING: ScreenGrab does not support 2D arrays, cubemaps, or mipmaps; only the first surface is written. Consider using DirectXTex instead.\n");
    //}

    //switch (desc.Format)
    //{
    //case DXGI_FORMAT_R32G32B32A32_FLOAT:            pfGuid = GUID_WICPixelFormat128bppRGBAFloat; break;
    //case DXGI_FORMAT_R16G16B16A16_FLOAT:            pfGuid = GUID_WICPixelFormat64bppRGBAHalf; break;
    //case DXGI_FORMAT_R16G16B16A16_UNORM:            pfGuid = GUID_WICPixelFormat64bppRGBA; break;
    //case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:    pfGuid = GUID_WICPixelFormat32bppRGBA1010102XR; break; // DXGI 1.1
    //case DXGI_FORMAT_R10G10B10A2_UNORM:             pfGuid = GUID_WICPixelFormat32bppRGBA1010102; break;
    //case DXGI_FORMAT_B5G5R5A1_UNORM:                pfGuid = GUID_WICPixelFormat16bppBGRA5551; break;
    //case DXGI_FORMAT_B5G6R5_UNORM:                  pfGuid = GUID_WICPixelFormat16bppBGR565; break;
    //case DXGI_FORMAT_R32_FLOAT:                     pfGuid = GUID_WICPixelFormat32bppGrayFloat; break;
    //case DXGI_FORMAT_R16_FLOAT:                     pfGuid = GUID_WICPixelFormat16bppGrayHalf; break;
    //case DXGI_FORMAT_R16_UNORM:                     pfGuid = GUID_WICPixelFormat16bppGray; break;
    //case DXGI_FORMAT_R8_UNORM:                      pfGuid = GUID_WICPixelFormat8bppGray; break;
    //case DXGI_FORMAT_A8_UNORM:                      pfGuid = GUID_WICPixelFormat8bppAlpha; break;

    //case DXGI_FORMAT_R8G8B8A8_UNORM:
    //    pfGuid = GUID_WICPixelFormat32bppRGBA;
    //    break;

    //case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    //    pfGuid = GUID_WICPixelFormat32bppRGBA;
    //    sRGB = true;
    //    break;

    //case DXGI_FORMAT_B8G8R8A8_UNORM: // DXGI 1.1
    //    pfGuid = GUID_WICPixelFormat32bppBGRA;
    //    break;

    //case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: // DXGI 1.1
    //    pfGuid = GUID_WICPixelFormat32bppBGRA;
    //    sRGB = true;
    //    break;

    //case DXGI_FORMAT_B8G8R8X8_UNORM: // DXGI 1.1
    //    pfGuid = GUID_WICPixelFormat32bppBGR;
    //    break;

    //case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: // DXGI 1.1
    //    pfGuid = GUID_WICPixelFormat32bppBGR;
    //    sRGB = true;
    //    break;

    //default:
    //    std::string msg = "ERROR: ScreenGrab does not support all DXGI formats (" + std::to_string(static_cast<uint32_t>(desc.Format)) + "). Consider using DirectXTex.";
    //    Debug(msg.c_str());
    //    hr = HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
    //}
    //-----------------------------------------


    if (FAILED(hr))
    {
        Debug("Save Failed:");
    }
    else
    {
        Debug("Save Succeeded:");
    }

    err = format_error(hr);
    Debug(err.c_str());
}

static void UNITY_INTERFACE_API OnRenderEvent(int eventID)
{
    //// Unknown / unsupported graphics device type? Do nothing
    //if (s_CurrentAPI == NULL)
    //    return;
    SaveImage();
}


// --------------------------------------------------------------------------
// GetRenderEventFunc, an example function we export which is used to get a rendering event callback function.

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
    return OnRenderEvent;
}
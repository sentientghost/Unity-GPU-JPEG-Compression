#include "pch.h"
#include "D3D11Plugin.h"


// Added requried namespaces
using namespace DirectX;
using namespace DirectX::LoaderHelpers;
using namespace Microsoft::WRL;
using Microsoft::WRL::ComPtr;

namespace DirectX
{
    extern bool _IsWIC2() noexcept;
    extern IWICImagingFactory* _GetWIC() noexcept;
}

namespace globals 
{
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
}

// UnitySetInterfaces global variables
static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
static UnityGfxRenderer s_RendererType = kUnityGfxRendererNull;

// Texture global variables
static ID3D11Resource* g_TextureHandle = NULL;
static float g_TextureQuality = 0;
IWICImagingFactory* pWIC;

D3D11_TEXTURE2D_DESC desc = {};
ComPtr<ID3D11Texture2D> pStaging;

ComPtr<IWICBitmapEncoder> encoder;
ComPtr<IWICBitmapFrameEncode> frame;
ComPtr<IWICStream> stream;
ComPtr<IPropertyBag2> props;

WICPixelFormatGUID pfGuid = {};
WICPixelFormatGUID targetGuid = {};

// Debug global variables
typedef void (*FuncPtr)(const char*);
FuncPtr Debug;

// General global variables
std::string dataPath = "";
float nativeTimes[3] = { 0, 0, 0 };


// --------------------------------------------------------------------------
// UNITY DEFINED FUNCTIONS

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

// Function to return time metrics to C# script
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API FillNativeTimes(float* data, int count)
{
    for (int i = 0; i < count; i++)
    {
        data[i] = nativeTimes[i];
    }
}

// Gets Unity Texture2D resource
extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(ID3D11Resource* textureHandle, int imageWidth, int imageHeight, float cameraQuality, char* path)
{
	g_TextureHandle = textureHandle;
    g_TextureQuality = cameraQuality;
    dataPath = path;

    std::string err = "";
    HRESULT hr = S_OK;
    std::wstring wstemp = StringToWideString(dataPath);
    LPCWSTR filePath = wstemp.c_str();

    // COPY
    auto startTime = std::chrono::high_resolution_clock::now();
    if (SUCCEEDED(hr))
        hr = CaptureTexture(globals::context, g_TextureHandle, desc, pStaging);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = endTime - startTime;
    nativeTimes[0] = duration.count();

    // ENCODE
    startTime = std::chrono::high_resolution_clock::now();
    if (SUCCEEDED(hr))
    {
        hr = EncodeTexture(globals::context, GUID_ContainerFormatJpeg, filePath, nullptr, 
            [&](IPropertyBag2* props)
            {
                PROPBAG2 option[1] = { 0 };
                option[0].pstrName = const_cast<wchar_t*>(L"ImageQuality");

                VARIANT varValue[1];
                varValue[0].vt = VT_R4;
                varValue[0].fltVal = g_TextureQuality;

                (void)props->Write(1, option, varValue);
            }, false);
    }
    endTime = std::chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    nativeTimes[1] = duration.count();

    // WRITE
    startTime = std::chrono::high_resolution_clock::now();
    if (SUCCEEDED(hr))
    {
        hr = SaveWICTextureToFile(filePath);
    }
    endTime = std::chrono::high_resolution_clock::now();
    duration = endTime - startTime;
    nativeTimes[2] = duration.count();

    if (FAILED(hr))
    {
        err = FormatError(hr);
        Debug(err.c_str());
    }
}


// --------------------------------------------------------------------------
// DIRECTX TOOLKIT ADAPTED FUNCTIONS

HRESULT CaptureTexture(_In_ ID3D11DeviceContext* pContext, _In_ ID3D11Resource* pSource, D3D11_TEXTURE2D_DESC& desc, ComPtr<ID3D11Texture2D>& pStaging) noexcept
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
        pStaging = pTexture;
    }
    else
    {
        // Otherwise, create a staging texture from the non-MSAA source
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

HRESULT EncodeTexture(ID3D11DeviceContext* pContext, REFGUID guidContainerFormat, const wchar_t* filePath, const GUID* targetFormat, std::function<void(IPropertyBag2*)> setCustomProps, bool forceSRGB)
{
    // Initialise variable for HRESULT
    HRESULT hr = S_OK;

    // Determine source format's WIC equivalent
    bool sRGB = forceSRGB;
    switch (desc.Format)
    {
        case DXGI_FORMAT_R32G32B32A32_FLOAT:            pfGuid = GUID_WICPixelFormat128bppRGBAFloat; break;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:            pfGuid = GUID_WICPixelFormat64bppRGBAHalf; break;
        case DXGI_FORMAT_R16G16B16A16_UNORM:            pfGuid = GUID_WICPixelFormat64bppRGBA; break;
        case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:    pfGuid = GUID_WICPixelFormat32bppRGBA1010102XR; break; // DXGI 1.1
        case DXGI_FORMAT_R10G10B10A2_UNORM:             pfGuid = GUID_WICPixelFormat32bppRGBA1010102; break;
        case DXGI_FORMAT_B5G5R5A1_UNORM:                pfGuid = GUID_WICPixelFormat16bppBGRA5551; break;
        case DXGI_FORMAT_B5G6R5_UNORM:                  pfGuid = GUID_WICPixelFormat16bppBGR565; break;
        case DXGI_FORMAT_R32_FLOAT:                     pfGuid = GUID_WICPixelFormat32bppGrayFloat; break;
        case DXGI_FORMAT_R16_FLOAT:                     pfGuid = GUID_WICPixelFormat16bppGrayHalf; break;
        case DXGI_FORMAT_R16_UNORM:                     pfGuid = GUID_WICPixelFormat16bppGray; break;
        case DXGI_FORMAT_R8_UNORM:                      pfGuid = GUID_WICPixelFormat8bppGray; break;
        case DXGI_FORMAT_A8_UNORM:                      pfGuid = GUID_WICPixelFormat8bppAlpha; break;

        case DXGI_FORMAT_R8G8B8A8_UNORM:
            pfGuid = GUID_WICPixelFormat32bppRGBA;
            break;

        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            pfGuid = GUID_WICPixelFormat32bppRGBA;
            sRGB = true;
            break;

        case DXGI_FORMAT_B8G8R8A8_UNORM: // DXGI 1.1
            pfGuid = GUID_WICPixelFormat32bppBGRA;
            break;

        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: // DXGI 1.1
            pfGuid = GUID_WICPixelFormat32bppBGRA;
            sRGB = true;
            break;

        case DXGI_FORMAT_B8G8R8X8_UNORM: // DXGI 1.1
            pfGuid = GUID_WICPixelFormat32bppBGR;
            break;

        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: // DXGI 1.1
            pfGuid = GUID_WICPixelFormat32bppBGR;
            sRGB = true;
            break;

        default:
            std::string err = "ERROR: ScreenGrab does not support all DXGI formats " + std::to_string(static_cast<uint32_t>(desc.Format)) + ". Consider using DirectXTex.\n";
            Debug(err.c_str());
            return HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
    }

    pWIC = _GetWIC();
    if (!pWIC)
        return E_NOINTERFACE;

    hr = pWIC->CreateStream(stream.GetAddressOf());
    if (FAILED(hr))
        return hr;

    hr = stream->InitializeFromFilename(filePath, GENERIC_WRITE);
    if (FAILED(hr))
        return hr;

    auto_delete_file_wic delonfail(stream, filePath);

    hr = pWIC->CreateEncoder(guidContainerFormat, nullptr, encoder.GetAddressOf());
    if (FAILED(hr))
        return hr;

    hr = encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache);
    if (FAILED(hr))
        return hr;
    
    hr = encoder->CreateNewFrame(frame.GetAddressOf(), props.GetAddressOf());
    if (FAILED(hr))
        return hr;

    if (targetFormat && memcmp(&guidContainerFormat, &GUID_ContainerFormatBmp, sizeof(WICPixelFormatGUID)) == 0 && _IsWIC2())
    {
        // Opt-in to the WIC2 support for writing 32-bit Windows BMP files with an alpha channel
        PROPBAG2 option = {};
        option.pstrName = const_cast<wchar_t*>(L"EnableV5Header32bppBGRA");

        VARIANT varValue;
        varValue.vt = VT_BOOL;
        varValue.boolVal = VARIANT_TRUE;
        std::ignore = props->Write(1, &option, &varValue);
    }

    if (setCustomProps)
    {
        setCustomProps(props.Get());
    }

    hr = frame->Initialize(props.Get());
    if (FAILED(hr))
        return hr;

    hr = frame->SetSize(desc.Width, desc.Height);
    if (FAILED(hr))
        return hr;

    hr = frame->SetResolution(96, 96);
    if (FAILED(hr))
        return hr;

    // Pick a target format
    if (targetFormat)
    {
        targetGuid = *targetFormat;
    }
    else
    {
        // Screenshots don't typically include the alpha channel of the render target
        switch (desc.Format)
        {
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            if (_IsWIC2())
            {
                targetGuid = GUID_WICPixelFormat96bppRGBFloat;
            }
            else
            {
                targetGuid = GUID_WICPixelFormat24bppBGR;
            }
            break;

        case DXGI_FORMAT_R16G16B16A16_UNORM: targetGuid = GUID_WICPixelFormat48bppBGR; break;
        case DXGI_FORMAT_B5G5R5A1_UNORM:     targetGuid = GUID_WICPixelFormat16bppBGR555; break;
        case DXGI_FORMAT_B5G6R5_UNORM:       targetGuid = GUID_WICPixelFormat16bppBGR565; break;

        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R16_FLOAT:
        case DXGI_FORMAT_R16_UNORM:
        case DXGI_FORMAT_R8_UNORM:
        case DXGI_FORMAT_A8_UNORM:
            targetGuid = GUID_WICPixelFormat8bppGray;
            break;

        default:
            targetGuid = GUID_WICPixelFormat24bppBGR;
            break;
        }
    }

    hr = frame->SetPixelFormat(&targetGuid);
    if (FAILED(hr))
        return hr;

    if (targetFormat && memcmp(targetFormat, &targetGuid, sizeof(WICPixelFormatGUID)) != 0)
    {
        // Requested output pixel format is not supported by the WIC codec
        return E_FAIL;
    }

    // Encode WIC metadata
    ComPtr<IWICMetadataQueryWriter> metawriter;
    if (SUCCEEDED(frame->GetMetadataQueryWriter(metawriter.GetAddressOf())))
    {
        PROPVARIANT value;
        PropVariantInit(&value);

        value.vt = VT_LPSTR;
        value.pszVal = const_cast<char*>("DirectXTK");

        if (memcmp(&guidContainerFormat, &GUID_ContainerFormatPng, sizeof(GUID)) == 0)
        {
            // Set Software name
            std::ignore = metawriter->SetMetadataByName(L"/tEXt/{str=Software}", &value);

            // Set sRGB chunk
            if (sRGB)
            {
                value.vt = VT_UI1;
                value.bVal = 0;
                std::ignore = metawriter->SetMetadataByName(L"/sRGB/RenderingIntent", &value);
            }
            else
            {
                // add gAMA chunk with gamma 1.0
                value.vt = VT_UI4;
                value.uintVal = 100000; // gama value * 100,000 -- i.e. gamma 1.0
                std::ignore = metawriter->SetMetadataByName(L"/gAMA/ImageGamma", &value);

                // remove sRGB chunk which is added by default.
                std::ignore = metawriter->RemoveMetadataByName(L"/sRGB/RenderingIntent");
            }
        }
        else
        {
            // Set Software name
            std::ignore = metawriter->SetMetadataByName(L"System.ApplicationName", &value);

            if (sRGB)
            {
                // Set EXIF Colorspace of sRGB
                value.vt = VT_UI2;
                value.uiVal = 1;
                std::ignore = metawriter->SetMetadataByName(L"System.Image.ColorSpace", &value);
            }
        }
    }

    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = pContext->Map(pStaging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr))
        return hr;

    uint64_t imageSize = uint64_t(mapped.RowPitch) * uint64_t(desc.Height);
    if (imageSize > UINT32_MAX)
    {
        pContext->Unmap(pStaging.Get(), 0);
        return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);
    }

    if (memcmp(&targetGuid, &pfGuid, sizeof(WICPixelFormatGUID)) != 0)
    {
        // Conversion required to write
        ComPtr<IWICBitmap> source;
        hr = pWIC->CreateBitmapFromMemory(desc.Width, desc.Height,
            pfGuid,
            mapped.RowPitch, static_cast<UINT>(imageSize),
            static_cast<BYTE*>(mapped.pData), source.GetAddressOf());
        if (FAILED(hr))
        {
            pContext->Unmap(pStaging.Get(), 0);
            return hr;
        }

        ComPtr<IWICFormatConverter> FC;
        hr = pWIC->CreateFormatConverter(FC.GetAddressOf());
        if (FAILED(hr))
        {
            pContext->Unmap(pStaging.Get(), 0);
            return hr;
        }

        BOOL canConvert = FALSE;
        hr = FC->CanConvert(pfGuid, targetGuid, &canConvert);
        if (FAILED(hr) || !canConvert)
        {
            pContext->Unmap(pStaging.Get(), 0);
            return E_UNEXPECTED;
        }

        hr = FC->Initialize(source.Get(), targetGuid, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeMedianCut);
        if (FAILED(hr))
        {
            pContext->Unmap(pStaging.Get(), 0);
            return hr;
        }

        WICRect rect = { 0, 0, static_cast<INT>(desc.Width), static_cast<INT>(desc.Height) };
        hr = frame->WriteSource(FC.Get(), &rect);
    }
    else
    {
        // No conversion required
        hr = frame->WritePixels(desc.Height,
            mapped.RowPitch, static_cast<UINT>(imageSize),
            static_cast<BYTE*>(mapped.pData));
    }

    pContext->Unmap(pStaging.Get(), 0);

    if (FAILED(hr))
        return hr;

    hr = frame->Commit();
    if (FAILED(hr))
        return hr;

    delonfail.clear();

    return S_OK;
}

HRESULT SaveWICTextureToFile(const wchar_t* filePath)
{
    // Initialise variable for HRESULT
    HRESULT hr = S_OK;
    auto_delete_file_wic delonfail(stream, filePath);

    hr = encoder->Commit();
    if (FAILED(hr))
        return hr;

    frame->Release();

    encoder->Release();

    delonfail.clear();

    return S_OK;
}


// --------------------------------------------------------------------------
// PLUG-IN SPECIFIC DEFINED FUNCTIONS

// Convert HRESULT to string
std::string FormatError(unsigned __int32 hr)
{
    std::stringstream ss;
    ss << "Error code = 0x" << std::hex << hr << std::endl;
    return ss.str();
}

// Convert string to wide string
std::wstring StringToWideString(const std::string& s)
{
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    wchar_t* buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}

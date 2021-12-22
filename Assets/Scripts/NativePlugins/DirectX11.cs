using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;


public class DirectX11 : MonoBehaviour
{
    // Editor Options
    [Header("Camera Settings")]
    [SerializeField] Camera cameraObject;

    RenderTexture rt;

    //the name of the DLL you want to load stuff from
    private const string pluginName = "DirectX11";


    // Access C++ function from C# script
    [DllImport(pluginName)]
    private static extern void SetTextureFromUnity(System.IntPtr texture, int w, int h);
    
    [DllImport(pluginName)]
    private static extern IntPtr GetRenderEventFunc();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void DebugDelegate(string str);

    static void CallBackFunction(string str) { Debug.Log(str); }

    [DllImport(pluginName)]
    public static extern void SetDebugFunction(IntPtr fp);


    // Use this for initialization
    IEnumerator Start () {
        DebugDelegate callback_delegate = new DebugDelegate(CallBackFunction);
        // Convert callback_delegate into a function pointer that can be
        // used in unmanaged code.
        IntPtr intptr_delegate =
            Marshal.GetFunctionPointerForDelegate(callback_delegate);
        // Call the API passing along the function pointer.
        SetDebugFunction(intptr_delegate);

        Debug.Log("Check 1");
        //cameraObject = gameObject.GetComponent<Camera>();
 
        if (cameraObject != null && cameraObject.pixelHeight > 0 && cameraObject.pixelWidth > 0)
        {
            Debug.Log("Check 2");
            CreateTextureAndPassToPlugin();
            yield return StartCoroutine("CallPluginAtEndOfFrames");
        }
    }

    private void CreateTextureAndPassToPlugin()
	{
		// Create a texture
        //rt = new RenderTexture(1280, 720, 24);
        // rt = new RenderTexture(1280, 720, 0, RenderTextureFormat.ARGB32);
        // rt.filterMode = FilterMode.Point;
        // rt.Create();

        // cameraObject.targetTexture = rt;
        // cameraObject.Render();

		Texture2D tex = new Texture2D(256,256,TextureFormat.ARGB32,false);
		// Set point filtering just so we can see the pixels clearly
		tex.filterMode = FilterMode.Point;
		// Call Apply() so it's actually uploaded to the GPU
		tex.Apply();

		// // Set texture onto our material
		// GetComponent<Renderer>().material.mainTexture = tex;

		// Pass texture pointer to the plugin
		//SetTextureFromUnity(rt.GetNativeTexturePtr(), rt.width, rt.height);
		SetTextureFromUnity(tex.GetNativeTexturePtr(), tex.width, tex.height);
        Debug.Log("Sent Texture");
	}

    private IEnumerator CallPluginAtEndOfFrames()
	{
		while (true) {
			// Wait until all frame rendering is done
            yield return new WaitForSeconds(5);
			yield return new WaitForEndOfFrame();

            //cameraObject.Render();

			// Issue a plugin event with arbitrary integer identifier.
			// The plugin can distinguish between different
			// things it needs to do based on this ID.
			// For our simple plugin, it does not matter which ID we pass here.
			GL.IssuePluginEvent(GetRenderEventFunc(), 1);
		}
	}
}

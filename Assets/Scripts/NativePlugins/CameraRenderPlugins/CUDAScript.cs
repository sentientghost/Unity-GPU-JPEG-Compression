using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Rendering;

public class CUDAScript : MonoBehaviour
{
    // Import DLL Functions
    // Access C++ function from C# script
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void DebugDelegate(string str);

    [DllImport("CUDA")]
    public static extern void SetDebugFunction(IntPtr fp);

    [DllImport("CUDA")]
    private static extern int SetTextureFromUnity(System.IntPtr texture, int imageWidth, int imageHeight);
    
    [DllImport("CUDA")]
    private static extern IntPtr GetRenderEventFunc();

    [DllImport("CUDA")]
    private static extern void GetErrorString(int error);

    // Global variables
    private Coroutine imageCoroutine;
    int error;


    /**** MONOBEHAVIOUR EVENT FUNCTIONS ****/

    // Called before the first frame update
    IEnumerator Start() 
    {
        // Link callback_delegate to DebugCallback function
        DebugDelegate callback_delegate = new DebugDelegate(DebugCallBack);
        // Convert callback_delegate into a function pointer that can be used in unmanaged code
        IntPtr intptr_delegate = Marshal.GetFunctionPointerForDelegate(callback_delegate);
        // Call the API passing along the function pointer
        SetDebugFunction(intptr_delegate);

        CreateTextureAndPassToPlugin();
		yield return StartCoroutine("CallPluginAtEndOfFrames");
    }

    private void CreateTextureAndPassToPlugin()
	{
		// Create a texture
		Texture2D tex = new Texture2D(256,256,TextureFormat.ARGB32,false);
		// Set point filtering just so we can see the pixels clearly
		tex.filterMode = FilterMode.Point;
		// Call Apply() so it's actually uploaded to the GPU
		tex.Apply();

		// Set texture onto our material
		// GetComponent<Renderer>().material.mainTexture = tex;

		// Pass texture pointer to the plugin
		error = SetTextureFromUnity(tex.GetNativeTexturePtr(), tex.width, tex.height);
        //GL.IssuePluginEvent(GetRenderEventFunc(), 1);

        // GetErrorString(error);
        // Debug.Log("CUDA Error: " + error + " (" + desc.ToString() + ")");
	}

    private IEnumerator CallPluginAtEndOfFrames()
	{
		while (true) {
			// Wait until all frame rendering is done
			yield return new WaitForEndOfFrame();

			// Issue a plugin event with arbitrary integer identifier.
			// The plugin can distinguish between different
			// things it needs to do based on this ID.
			// For our simple plugin, it does not matter which ID we pass here.
			GL.IssuePluginEvent(GetRenderEventFunc(), 1);
		}
	}

    /**** USER DEFINED FUNCTIONS ****/
    static void DebugCallBack(string str) 
    { 
            Debug.Log(str);
    }
}

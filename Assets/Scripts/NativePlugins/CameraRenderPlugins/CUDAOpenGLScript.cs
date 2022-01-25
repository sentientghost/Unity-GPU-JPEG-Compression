using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Rendering;

public class CUDAOpenGLScript : MonoBehaviour
{
    // Import DLL Functions
    // Access C++ function from C# script
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void DebugDelegate(string str);

    [DllImport("CUDA_OpenGL_Interop")]
    public static extern void SetDebugFunction(IntPtr fp);

    [DllImport("CUDA_OpenGL_Interop")]
    private static extern void SetTextureFromUnity(System.IntPtr texture, int imageWidth, int imageHeight, int cameraQuality, string path);

    [DllImport("CUDA_OpenGL_Interop")]
    private static extern IntPtr GetRenderEventFunc();

    [DllImport("CUDA_OpenGL_Interop")]
    private static extern int GetCopyTime();

    [DllImport("CUDA_OpenGL_Interop")]
    private static extern int GetEncodeTime();

    [DllImport("CUDA_OpenGL_Interop")]
    private static extern int GetWriteTime();


    // Global variables
    private Coroutine imageCoroutine;
    Texture2D imageTexture;
    float[] times = new float[4] {0,0,0,0};
    string buildMode;


    /**** MONOBEHAVIOUR EVENT FUNCTIONS ****/

    // Called before the first frame update
    void Start() 
    {
        // Check which build mode Unity is running in
        if (Application.isEditor)
        {
            buildMode = "Editor";
        }
        else if (Application.isBatchMode)
        {
            buildMode = "Batch";
        }
        else
        {
            buildMode = "Windowed";
        }

        // Link debug_callback_delegate to DebugCallback function
        DebugDelegate debug_callback_delegate = new DebugDelegate(DebugCallBack);
        // Convert debug_callback_delegate into a function pointer that can be used in unmanaged code
        IntPtr debug_intptr_delegate = Marshal.GetFunctionPointerForDelegate(debug_callback_delegate);
        // Call the API passing along the function pointer
        SetDebugFunction(debug_intptr_delegate);

        // Initialize the image texture as 144p
        imageTexture = new Texture2D(256, 144, TextureFormat.ARGB32, false);
    }


    /**** USER DEFINED FUNCTIONS ****/

    // Function to start the coroutine
    public float[] CallTakeImage(int imageWidth, int imageHeight, Camera cameraObject, int cameraQuality, int frameCount) 
    {
        // Start "Take Image" coroutine
        imageCoroutine = StartCoroutine(TakeImage(imageWidth, imageHeight, cameraObject, cameraQuality, frameCount));

        // Return the image times
        return times;
    }

    // Coroutine function to take image from the screen
    IEnumerator TakeImage(int imageWidth, int imageHeight, Camera cameraObject, int cameraQuality, int frameCount)
    {
        // Read the screen buffer after rendering is complete
        yield return new WaitForEndOfFrame();

        // Create time variables
        float startTime = 1.0f;
        float endTime = 1.0f;

        // Create a texture in RGB24 format with the specified width and height
        // Set point filtering just so we can see the pixels clearly
        // Call Apply() so it's actually uploaded to the GPU
        UnityEngine.Object.Destroy(imageTexture);
        imageTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.ARGB32, false);
		imageTexture.filterMode = FilterMode.Point;
		imageTexture.Apply();

        // RENDER
        // Render the camera's view
        // The camera will send OnPreCull, OnPreRender and OnPostRender
        // OnPreCull - Event function that Unity calls before a Camera culls the scene
        // OnPreRender - Event function that Unity calls before a Camera renders the scene
        // OnPostRender - Event function that Unity calls after a Camera renders the scene
        startTime = Time.realtimeSinceStartup;
        cameraObject.Render();
        endTime = Time.realtimeSinceStartup;
        times[0] = ((endTime - startTime) * 1000);

        // READ/COPY
        // GPU to GPU
        // Read the active render texture into the image texture 
        startTime = Time.realtimeSinceStartup;
        Graphics.CopyTexture(cameraObject.activeTexture, imageTexture);
        endTime = Time.realtimeSinceStartup;
        times[1] = ((endTime - startTime) * 1000);

        // Get Application Datapath
        string path = ImageName(imageHeight, cameraQuality, frameCount);
		
        // ENCODE/COMPRESS and WRITE/SAVE
        // Encode the texture in JPG format and Write it to a file
        // Pass texture pointer to the plugin and call native plugin from the render thread
        // Issue the plugin event to copy, encode and write the image
		SetTextureFromUnity(imageTexture.GetNativeTexturePtr(), imageWidth, imageHeight, cameraQuality, path);
        GL.IssuePluginEvent(GetRenderEventFunc(), 1);
        
        // Acquire times from CUDA OpenGL Interop Plugin
        FillTimes();
    }

    // Function to return the filepath with an appropriate image name
    string ImageName(int imageHeight, int cameraQuality, int frameCount)
    {
        // Check the build mode of Unity
        if (buildMode == "Editor")
        {
            // Return filepath with appropriate image name for editor mode
            return string.Format("{0}/../Images/Native Plugins/CUDA OpenGL Interop/{1} Mode/{2} Scene/cudaGL_{3}p_{4}_{5}.jpg", Application.dataPath, buildMode, SceneManager.GetActiveScene().name, imageHeight, cameraQuality, frameCount+1);
        }   
        else
        {
            // Return filepath with appropriate image name for windowed and batch mode
            return string.Format("{0}/../../../../Images/Native Plugins/CUDA OpenGL Interop/{1} Mode/{2} Scene/cudaGL_{3}p_{4}_{5}.jpg", Application.dataPath, buildMode, SceneManager.GetActiveScene().name, imageHeight, cameraQuality, frameCount+1);
        }
    }

    // Fill times metrics with native times values from CUDA OpenGL Interop Plugin
    public void FillTimes()
	{
        // Fill Copy Time from CUDA OpenGL Interop Plugin
        times[1] += GetCopyTime()/10000f;

        // Fill Encode Time from CUDA OpenGL Interop Plugin
        times[2] = GetEncodeTime()/10000f;

        // Fill Write Time from CUDA OpenGL Interop Plugin
        times[3] = GetWriteTime()/10000f;
	}


    /**** STATIC DEFINED FUNCTIONS ****/

    // Function to log debug messages from CUDA OpenGL Interop Plugin
    static void DebugCallBack(string str) 
    { 
        Debug.Log(str);
    }
}

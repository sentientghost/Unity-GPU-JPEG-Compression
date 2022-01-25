using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Rendering;


public class CoroutinesScript : MonoBehaviour 
{
    // Global variables
    private Coroutine imageCoroutine;
    Texture2D imageTexture;
    float[] times = new float[4];
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
        UnityEngine.Object.Destroy(imageTexture);
        imageTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.ARGB32, false);

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

        // The Render Texture in RenderTexture.active is the one that will be read by ReadPixels
        RenderTexture.active = cameraObject.targetTexture;

        // READ/COPY
        // GPU to CPU
        // Read the active render texture into the image texture (from screen to image texture)
        startTime = Time.realtimeSinceStartup;
        imageTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        endTime = Time.realtimeSinceStartup;
        times[1] = ((endTime - startTime) * 1000);

        // ENCODE/COMPRESS
        // Encode the texture in JPG format
        startTime = Time.realtimeSinceStartup;
        byte[] bytes = imageTexture.EncodeToJPG(cameraQuality);
        endTime = Time.realtimeSinceStartup;
        times[2] = ((endTime - startTime) * 1000);

        // WRITE/SAVE
        // Write the returned byte array to a file
        string filename = ImageName(imageHeight, cameraQuality, frameCount);
        startTime = Time.realtimeSinceStartup;
        System.IO.File.WriteAllBytes(filename, bytes);
        endTime = Time.realtimeSinceStartup;
        times[3] = ((endTime - startTime) * 1000);
    }

    // Function to return the filepath with an appropriate image name
    string ImageName(int imageHeight, int cameraQuality, int frameCount)
    {
        // Check the build mode of Unity
        if (buildMode == "Editor")
        {
            // Return filepath with appropriate image name for editor mode
            return string.Format("{0}/../Images/Current Performance/{1} Mode/{2} Scene/coroutines_{3}p_{4}_{5}.jpg", Application.dataPath, buildMode, SceneManager.GetActiveScene().name, imageHeight, cameraQuality, frameCount+1);
        }   
        else
        {
            // Return filepath with appropriate image name for windowed and batch mode
            return string.Format("{0}/../../../../Images/Current Performance/{1} Mode/{2} Scene/coroutines_{3}p_{4}_{5}.jpg", Application.dataPath, buildMode, SceneManager.GetActiveScene().name, imageHeight, cameraQuality, frameCount+1);
        }
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;


public class LinearScript : MonoBehaviour 
{
    public float[] CallTakeImage(int imageWidth, int imageHeight, Camera cameraObject, int cameraQuality)
    {
        // Create time variables
        float startTime = 1.0f;
        float endTime = 1.0f;
        float[] times = new float[4];

        // Create a texture in RGB24 format with the specified width and height
        Texture2D image = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

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
        // Read the active render texture into the image texture (from screen to image texture)
        startTime = Time.realtimeSinceStartup;
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        endTime = Time.realtimeSinceStartup;
        times[1] = ((endTime - startTime) * 1000);

        // ENCODE/COMPRESS
        // Encode the texture in JPG format
        startTime = Time.realtimeSinceStartup;
        byte[] bytes = image.EncodeToJPG(cameraQuality);
        endTime = Time.realtimeSinceStartup;
        times[2] = ((endTime - startTime) * 1000);

        // WRITE/SAVE
        // Write the returned byte array to a file
        string filename = ImageName(imageWidth, imageHeight);
        startTime = Time.realtimeSinceStartup;
        System.IO.File.WriteAllBytes(filename, bytes);
        endTime = Time.realtimeSinceStartup;
        times[3] = ((endTime - startTime) * 1000);

        // Return performance metrics
        // returns the render, copy, encode and write times
        return times;
    }

    string ImageName(int imageWidth, int imageHeight)
    {
        return string.Format("{0}/../Images/image_{1}x{2}_{3}.jpg", Application.dataPath, imageWidth, imageHeight, System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss.fff"));
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;


public class CameraScript : MonoBehaviour 
{
    enum Resolutions { _1440p = 1440, _1080p = 1080, _720p = 720, _480p = 480, _360p = 360, _240p = 240, _144p = 144 }

    [Header("Camera Settings")]
    [SerializeField] Camera cameraObject;
    [SerializeField] int cameraQuality = 75;
    [SerializeField] int cameraFrequency = 25;
    [SerializeField] float cameraTime = 10.0f;

    [Header("Resolution")]
    [SerializeField] Resolutions resolution = Resolutions._720p;
    int imageWidth;
    int imageHeight;

    float intervalTime;
    float timeElapsed;
    float[] renderTimes = new float[3];
    float[] times = new float[4];
    int imageCountIdeal = 0;
    int imageCountActual = 0;

    void Awake() 
    {
        if (cameraObject.targetTexture == null)
        {
            float aspectRatio = 16.0f / 9.0f;
            imageHeight = (int)resolution;
            imageWidth = (int)(imageHeight * aspectRatio);
            cameraObject.targetTexture = new RenderTexture(imageWidth, imageHeight, 24);
        }
        else
        {
            imageWidth = cameraObject.targetTexture.width;
            imageHeight = cameraObject.targetTexture.height;
        }

        cameraObject.gameObject.SetActive(false);
    }

    void Start() 
    {
        intervalTime = Time.fixedTime + (1.0f/cameraFrequency);

        // Add your callback to the delegate's invocation list
        Camera.onPreCull += OnPreCullCallback;
        Camera.onPreRender += OnPreRenderCallback;
        Camera.onPostRender += OnPostRenderCallback;
    }

    void Update() 
    {
        timeElapsed += Time.deltaTime;

        if (cameraTime > 0)
        {
            if (timeElapsed >= cameraTime)
            {
                EditorApplication.isPlaying = false;
            }
        }
    }

    void FixedUpdate() {
        if (Time.fixedTime >= intervalTime) {
            CallTakeImage();
            imageCountActual += 1;
            intervalTime = Time.fixedTime + (1.0f/cameraFrequency);
        }
    }
    
    void OnPreCullCallback(Camera cam) 
    {
        if (cam == cameraObject)
        {
            renderTimes[0] = Time.realtimeSinceStartup;
        }
    }

    void OnPreRenderCallback(Camera cam) 
    {
        if (cam == cameraObject)
        {
            renderTimes[1] = Time.realtimeSinceStartup;
        }
    }

    void OnPostRenderCallback(Camera cam) 
    {
        if (cam == cameraObject)
        {
            renderTimes[2] = Time.realtimeSinceStartup;
        }
    }

    void CallTakeImage()
    {
        // Create time variables
        float startTime = 1.0f;
        float endTime = 1.0f;

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
        times[0] += ((endTime - startTime) * 1000);

        // The Render Texture in RenderTexture.active is the one that will be read by ReadPixels
        RenderTexture.active = cameraObject.targetTexture;

        // READ/COPY
        // Read the active render texture into the image texture (from screen to image texture)
        startTime = Time.realtimeSinceStartup;
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        endTime = Time.realtimeSinceStartup;
        times[1] += ((endTime - startTime) * 1000);

        // ENCODE/COMPRESS
        // Encode the texture in JPG format
        startTime = Time.realtimeSinceStartup;
        byte[] bytes = image.EncodeToJPG(cameraQuality);
        endTime = Time.realtimeSinceStartup;
        times[2] += ((endTime - startTime) * 1000);

        // WRITE/SAVE
        // Write the returned byte array to a file
        string filename = ImageName();
        startTime = Time.realtimeSinceStartup;
        System.IO.File.WriteAllBytes(filename, bytes);
        endTime = Time.realtimeSinceStartup;
        times[3] += ((endTime - startTime) * 1000);
    }

    string ImageName()
    {
        return string.Format("{0}/../Images/image_{1}x{2}_{3}.jpg", Application.dataPath, imageWidth, imageHeight, System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss.fff"));
    }

    void OnDisable()
    {
        if (cameraTime > 0)
        {
            imageCountIdeal = (int)(cameraTime * cameraFrequency);
        }
        else
        {
            imageCountIdeal = (int)(timeElapsed * cameraFrequency);
        }
        
        float cullTime = renderTimes[1] - renderTimes[0];
        float renderTime = renderTimes[2] - renderTimes[1];

        // Calculate Average
        times[0] = times[0] / imageCountActual;
        times[1] = times[1] / imageCountActual;
        times[2] = times[2] / imageCountActual;
        times[3] = times[3] / imageCountActual;

        Debug.Log("Time Elasped: " + timeElapsed);
        Debug.Log("Image Count: " + imageCountIdeal + " (ideal), " + imageCountActual + " (actual)");
        //Debug.Log("PreCull: " + renderTimes[0] + ", PreRender: " + renderTimes[1] + ", PostRender: " + renderTimes[2]);
        //Debug.Log("Cull Time: " + cullTime + ", Render Time: " + renderTime);
        Debug.Log("Render: " + times[0] + " ms, Read/Copy: " + times[1] + " ms, Encode/Compress: " + times[2] + " ms, Write/Save: " + times[3] + " ms");
    }

    void OnDestroy()
    {
        // Remove your callback from the delegate's invocation list
        Camera.onPreCull -= OnPreCullCallback;
        Camera.onPreRender -= OnPreRenderCallback;
        Camera.onPostRender -= OnPostRenderCallback;
    }
}

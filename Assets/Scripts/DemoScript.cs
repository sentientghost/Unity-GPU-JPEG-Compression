using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

public class DemoScript : MonoBehaviour
{
    // Enums
    public enum Solution 
    {
        [InspectorName("Linear")] Linear, 
        [InspectorName("Coroutines")] Coroutines, 
        [InspectorName("DirectX 11")] DirectX11, 
        [InspectorName("CUDA OpenGL Interop")] CUDAOpenGLInterop
    };

    public enum ImageResolution
    {
        [InspectorName("144p")] _144 = 144, 
        [InspectorName("240p")] _240 = 240, 
        [InspectorName("360p")] _360 = 360, 
        [InspectorName("480p")] _480 = 480,
        [InspectorName("720p")] _720 = 720, 
        [InspectorName("1080p")] _1080 = 1080, 
        [InspectorName("1440p")] _1440 = 1440
    };

    public enum JPEGQuality 
    {
        [InspectorName("75%")] _75 = 75, 
        [InspectorName("80%")] _80 = 80, 
        [InspectorName("85%")] _85 = 85, 
        [InspectorName("90%")] _90 = 90,
        [InspectorName("95%")] _95 = 95, 
        [InspectorName("100%")] _100 = 100 
    };

    // Editor Options
    [Header("Camera Settings")]
    [SerializeField] Camera cameraObject;
    public int cameraFrequency = 25;
    public ImageResolution imageResolution = ImageResolution._720;

    [Header("Compression Settings")]
    public Solution solution = Solution.Linear;
    public JPEGQuality jpegQuality = JPEGQuality._75;

    [Header("Script Settings")]
    public bool verbose = false;
    public string outputDirectory = "";
    
    // Configuration Variables
    string imagePath;
    float intervalTime;
    float timeElapsed;
    int[] imageWH;

    // Script References
    LinearScript linearScript;
    CoroutinesScript coroutinesScript;
    CUDAOpenGLScript cudaOpenGLScript;
    DirectX11Script directX11Script;


    /**** MONOBEHAVIOUR EVENT FUNCTIONS ****/

    // Called when the scene starts but before the start function
    void Awake() 
    {   
        // Set correct Graphics API based on Solution
        if (solution == Solution.DirectX11)
        {
            var apis = new GraphicsDeviceType[] {GraphicsDeviceType.Direct3D11};
            PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64, apis);
        }
        else if (solution == Solution.CUDAOpenGLInterop)
        {
            var apis = new GraphicsDeviceType[] {GraphicsDeviceType.OpenGLCore};
            PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64, apis);
        }

        if (outputDirectory == "")
        {
            outputDirectory = Application.dataPath + "/../Images";

            Debug.LogWarning("Output Directory not set. Defaulting to: " + outputDirectory);
        }

        if (verbose)
        {
            GraphicsDeviceType[] graphicsAPI = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64);
            Debug.Log("Graphics API: " + graphicsAPI[0]);
        }
    }

    // Called before the first frame update
    void Start()
    {
        // Initialize the correct solution and camera offset angle
        switch(solution)
        {
            case Solution.Linear:
                linearScript = gameObject.GetComponent<LinearScript> ();
                break;

            case Solution.Coroutines:
                coroutinesScript = gameObject.GetComponent<CoroutinesScript> ();
                break;

            case Solution.DirectX11:
                directX11Script = gameObject.GetComponent<DirectX11Script> ();
                Vector3 directX11Rotation = new Vector3 (0, 0, 180);
                cameraObject.transform.eulerAngles = cameraObject.transform.eulerAngles - directX11Rotation;
                break;

            case Solution.CUDAOpenGLInterop:
                cudaOpenGLScript = gameObject.GetComponent<CUDAOpenGLScript> ();
                Vector3 cudaOpenGLRotation = new Vector3 (0, 0, 180);
                cameraObject.transform.eulerAngles = cameraObject.transform.eulerAngles - cudaOpenGLRotation;
                break;

            default:
                Debug.LogError("Error: Unsupported Solution Selected");
                UnityEditor.EditorApplication.isPlaying = false;
                break;
        }

        // Initialize Render Texture for Camera
        CalculateImageSize((int) imageResolution);

        // Initialize the interval time based on camera frequency
        intervalTime = Time.fixedTime + (1.0f/cameraFrequency);

        if (verbose)
        {
            Debug.Log("Solution: " + solution);
            Debug.Log("Camera Frequency: " + cameraFrequency + " Hz, Interval Time: " + (intervalTime*1000) + " ms");
        }
    }

    // Called once per frame (main function for frame updates)
    void Update()
    {
        // Update the time elapsed since the start of the scene
        timeElapsed += Time.deltaTime;
    }

    // Called on a timer (every 0.02s) which is independent of frame rate
    void FixedUpdate() 
    {
        // Check if the time condition is true which is constraint by the camera frequency
        if (Time.fixedTime >= intervalTime)
        {   
            string dateTime = System.DateTime.Now.ToString("dd/MM/yyyy-hh:mm:ss");
            string scene = SceneManager.GetActiveScene().name;
            imagePath = outputDirectory + "/" + solution + "_" + scene + "_" + dateTime + ".jpg";
            SaveScreenJPG();
        }
    }


    /**** USER DEFINED FUNCTIONS ****/

    // Function to calculate width and height based on resolution and aspect ratio
    void CalculateImageSize(int resolution)
    {
        // Define aspect ratio and calculate the new width and height
        float aspectRatio = 16.0f / 9.0f;
        imageWH = new int[2] {(int)(resolution * aspectRatio), resolution};

        // Initialize render texture based on calculated width and height
        cameraObject.targetTexture = new RenderTexture(imageWH[0], imageWH[1], 24, RenderTextureFormat.ARGB32);
    }

    // Function to take image from the screen
    bool SaveScreenJPG()
    {
        // Initialize image times and success flag for the execution time metrics  
        bool success = true;
        float[] imageTimes = new float[4];
        
        // Save JPEG Screenshot using the relevant Solution Script and save the image times
        switch(solution)
        {
            case Solution.Linear:
                imageTimes = linearScript.CallTakeImage(imageWH[0], imageWH[1], cameraObject, (int) jpegQuality, imagePath);
                break;

            case Solution.Coroutines:
                imageTimes = coroutinesScript.CallTakeImage(imageWH[0], imageWH[1], cameraObject, (int) jpegQuality, imagePath);
                break;

            case Solution.DirectX11:
                imageTimes = directX11Script.CallTakeImage(imageWH[0], imageWH[1], cameraObject, (int) jpegQuality, imagePath);
                break;

            case Solution.CUDAOpenGLInterop:
                imageTimes = cudaOpenGLScript.CallTakeImage(imageWH[0], imageWH[1], cameraObject, (int) jpegQuality, imagePath);
                break;

            default:
                Debug.LogError("Error: Unsupported Solution Script");
                UnityEditor.EditorApplication.isPlaying = false;
                break;
        }

        // Check if the scipts ran successfully based on the image times 
        if (imageTimes[0] == 0 || imageTimes[1] == 0 || imageTimes[2] == 0 || imageTimes[3] == 0)
        {
            // Take image failed so return false
            success = false;
        }
        else if (verbose)
        {
            Debug.Log("RET: " + imageTimes[0] + " ms, CET: " + imageTimes[1] + " ms, EET: " + imageTimes[2] + " ms, WET: " + imageTimes[3] + " ms");
        }

        return success;
    }
}

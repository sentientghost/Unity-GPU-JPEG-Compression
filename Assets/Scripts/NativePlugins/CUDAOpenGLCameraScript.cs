using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Rendering;

public class CUDAOpenGLCameraScript : MonoBehaviour
{
    // Editor Options
    [Header("Camera Settings")]
    [SerializeField] Camera cameraObject;
    [Header("Ball Settings")]
    public GameObject ballPrefab;

    // Flags to track progress of code
    bool bufferFlag = true;
    bool exitFlag = false;
    bool testFlag = true;
    bool checkFlag = true;

    // Test parameters
    int cameraFrequency = 25;
    int bufferTime = 2; 
    int bufferCount = 0;
    int frameCount;
    int imageCount;
    int apiCount;
    int resolutionCount;
    int qualityCount;
    string buildMode;
    int[] resolutions;
    int[] qualities;
    float intervalTime;
    float timeElapsed;
    float bufferTimeElapsed;
    int[] imageSize = new int[2];
    GameObject ballObject;

    // Script References
    CUDAOpenGLScript cudaOpenGLScript;

    // Scene metrics such that:
    // 1st bracket is graphics APIs (DirectX 11 OR OpenGL Core)
    // 2nd bracket is resolutions (144, 240, 360, 480, 720, 1080, 1440)
    // 3rd bracket is jpeg qualities (75, 80, 85, 90, 95, 100)
    // 4th bracket is performance metrics (render, copy, encode, write)
    // 5th bracket is raw data for that performance metric (250 samples)
    float[,,,,] sceneMetrics = new float[1, 7, 6, 4, 250];


    /**** MONOBEHAVIOUR EVENT FUNCTIONS ****/

    // Called when the scene starts but before the start function
    void Awake() 
    {
        // Define the image resolutions and JPEG qualities 
        resolutions = new int[] {144, 240, 360, 480, 720, 1080, 1440};
        qualities = new int[] {75, 80, 85, 90, 95, 100};

        // Initialize the first render texture and set it to the camera target texture
        imageSize = CalculateImageSize(resolutions[0]);
    }

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

        // Initialize CUDA OpenGL Script
        cudaOpenGLScript = gameObject.GetComponent<CUDAOpenGLScript> ();
        
        // Initialize ball reference for the scene
        string objectName = SceneManager.GetActiveScene().name + "SceneBall";
        ballObject = GameObject.Find(objectName);
        
        // Initialize counts to zero
        frameCount = 0;
        imageCount = 0;
        apiCount = 0; 
        resolutionCount = 0; 
        qualityCount = 0; 

        // Initialize the interval time based on camera frequency
        intervalTime = Time.fixedTime + (1.0f/cameraFrequency);

        // Pause the game physics
        Physics.autoSimulation = false;
    }

    // Called once per frame (main function for frame updates)
    void Update() 
    {
        // Update the time elapsed since the start of the scene
        timeElapsed += Time.deltaTime;

        // Check if exit condition is true
        if (SceneManager.GetActiveScene().buildIndex == 3 && exitFlag == true)
        {
            // Conditional compilation to close Unity
            #if UNITY_EDITOR
            EditorApplication.isPlaying = false;
            #else
            Application.Quit();
            #endif
        }
        else if (exitFlag == true)
        {
            // Unloads current scene and loads the next scene in the build index
            SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex + 1, LoadSceneMode.Single);
        }
    }

    // Called on a timer (every 0.02s) which is independent of frame rate
    void FixedUpdate() 
    {
        // Check if the time condition is true which is constraint by the camera frequency
        if (Time.fixedTime >= intervalTime)
        {   
            // Check if it is buffer time or test time 
            if (bufferFlag == true)
            {
                // Check if buffer time is finished
                if (bufferCount == (bufferTime * cameraFrequency))
                {
                    // Reset buffer time and enable physics
                    bufferCount = 0;
                    bufferFlag = false;

                    Physics.autoSimulation = true;

                    Debug.Log("Buffer Time Elasped: " + (timeElapsed - bufferTimeElapsed) + " s");
                }
                else if (bufferCount == 0)
                {
                    // Instantiate ball prefab and disable physics
                    bufferTimeElapsed = timeElapsed;

                    Destroy(ballObject);
                    ballObject = Instantiate(ballPrefab);

                    Physics.autoSimulation = false;

                    bufferCount += 1;
                }
                else
                {
                    // Increment the buffer count
                    bufferCount += 1;
                }
            }
            else
            {
                // Check if testing is complete
                if (testFlag)
                {
                    // Check if 250 frames have elasped
                    if (frameCount >= 250)
                    {
                        // Reset frame count and increment to next image
                        frameCount = 0;
                        imageCount += 1;

                        // Update progress and output logs
                        float progress = 100 * ((imageCount*250 + frameCount) / 10500f);
                        Debug.Log("Progress: " + progress.ToString("F2") + " %");
                        Debug.Log("Current Count Value: " + apiCount + " " + resolutionCount + " " + qualityCount);

                        // Increment counts to start next test case
                        IncrementCounts();
                    }
                    else if (!checkFlag)
                    {
                        // Retake image if it was previously unsuccessful
                        frameCount -= 1;

                        // Ensure frameCount stays within index range for the arrays
                        if (frameCount < 0)
                            frameCount = 0;

                        checkFlag = SaveScreenJPG();
                        frameCount += 1;
                    }
                    else
                    {
                        // Take image and record if it was successful or not
                        checkFlag = SaveScreenJPG();
                        frameCount += 1;
                    }
                }
                else
                {
                    // Output csv files 
                    OutputMetrics();
                }
            }
            
            // Update the interval time
            intervalTime = Time.fixedTime + (1.0f/cameraFrequency);
        }
    }

    // Called when the behaviour becomes disabled or inactive
    void OnDisable()
    {
        // Output logs when scene is completed
        Debug.Log("Total Time Elasped: " + timeElapsed + " s");
        Debug.Log("Total Images Saved: " + (imageCount*250 + frameCount));
        Debug.Log("Success!!! Test Completed");
    }


    /**** USER DEFINED FUNCTIONS ****/

    // Function to calculate width and height based on resolution and aspect ratio
    int[] CalculateImageSize(int resolution)
    {
        // Define aspect ratio and calculate the new width and height
        float aspectRatio = 16.0f / 9.0f;
        int[] imageWH = new int[2] {(int)(resolution * aspectRatio), resolution};

        // Destroy previous targetTexture and initialize new render texture based on calculated width and height
        UnityEngine.Object.Destroy(cameraObject.targetTexture);
        cameraObject.targetTexture = new RenderTexture(imageWH[0], imageWH[1], 24, RenderTextureFormat.ARGB32);

        // return int[] with width and height
        return imageWH;
    }

    // Function to increment between different test cases using various counts
    void IncrementCounts()
    {
        // ONLY 1 GRAPHICS API CAN BE TESTED AT A TIME
        // Check if test case is complete based on count values
        if (apiCount == 0 && resolutionCount == 6 && qualityCount == 5)
        {
            testFlag = false;
            return;
        }

        // Increment to next test case
        qualityCount += 1;
        if (qualityCount >= 6)
        {
            qualityCount = 0;
            resolutionCount += 1;

            if (resolutionCount >= 7)
            {
                apiCount += 1;
                resolutionCount = 0;
            }
        }

        // Enable buffer time
        bufferFlag = true;
    }

    // Function to take image from the screen
    bool SaveScreenJPG ()
    {
        // Initialize image times and the image width and height
        float[] imageTimes = new float[4];
        int[] imageWH = CalculateImageSize(resolutions[resolutionCount]);

        // Check which script to run based on the code count
        if (apiCount == 0)
        {
            //Take image using the CUDA OpenGL script and save the image times
            imageTimes = cudaOpenGLScript.CallTakeImage(imageWH[0], imageWH[1], cameraObject, qualities[qualityCount], frameCount);
        }

        // Check if the scipts ran successfully based on the image times 
        if (imageTimes[0] == 0 || imageTimes[1] == 0 || imageTimes[2] == 0 || imageTimes[3] == 0)
        {
            // Take image failed so return false
            return false;
        }
        else
        {
            // Save image times to scene metrics
            for (int metricCount = 0; metricCount <= 3; metricCount++)
            {
                sceneMetrics[apiCount, resolutionCount, qualityCount, metricCount, frameCount] = imageTimes[metricCount];
            }

            // Take is succeeded so return true
            return true;
        }
    }

    // Function to process the data from the float[] in scene metrics to string in csv data
    string ProcessData()
    {
        // Initialize string builder
        StringBuilder csvData = new StringBuilder();

        // Initialize column heading options
        string[] graphicsAPI = new string[1] {"cudaGL"};        
        string[] imageResolution = new string[7] {"144p", "240p", "360p", "480p", "720p", "1080p", "1440p"};
        string[] qualityLevel = new string[6] {"75", "80", "85", "90", "95", "100"};
        string[] metric = new string[4] {"render", "copy", "encode", "write"};
        
        // loop to iterate through each frame (250)
        for (int i = -1; i < 250; i++)
        {
            // loop to iterate through each graphics API (1)
            for(int j = 0; j < 1; j++)
            {
                // loop to iterate through each image resolution (7)
                for(int k = 0; k < 7; k++)
                {
                    // loop to iterate through each JPEG quality (6)
                    for(int l = 0; l < 6; l++)
                    {
                        // loop to iterate through each image metric (4)
                        for(int m = 0; m < 4; m++)
                        {
                            // Check if it is the first row of the csv file
                            if (i == -1)
                            {
                                // Add column headings to csv file
                                string columnHeading = graphicsAPI[j] + "_" + imageResolution[k] + "_" + qualityLevel[l] + "_" + metric[m] + ",";
                                csvData.Append(columnHeading);
                            }
                            else
                            {
                                // Add scene metrics to csv file
                                csvData.Append(sceneMetrics[j, k, l, m, i].ToString()).Append(",");
                            }
                        }
                    }
                }
            }
            // Start new data on a next row
            csvData.Append("\n");
        } 

        // Return csv data as a string
        return csvData.ToString();
    }

    // Function to determine the correct filepath based on build mode and scene type 
    string FilePath()
    {
        // Check the build mode of Unity
        if (buildMode == "Editor")
        {
            // Return filepath for editor mode OpenGL Core graphics API
            return string.Format("{0}/../Metrics/Native Plugins/CUDA OpenGL Interop/{1}-Mode_{2}-Scene.csv", Application.dataPath, buildMode, SceneManager.GetActiveScene().name);
        }
        else
        {
            // Return filepath for windowed and batch mode OpenGL Core graphics API
            return string.Format("{0}/../../../Metrics/Native Plugins/CUDA OpenGL Interop/{1}-Mode_{2}-Scene.csv", Application.dataPath, buildMode, SceneManager.GetActiveScene().name);
        } 
    }

    // Function to save the scene metrics as a csv file
    void OutputMetrics()
    {
        // Initialize the csv data and file path
        string csvData = ProcessData();
        string filePath = FilePath();

        // Save/Write csv data to the filepath and output the logs
        System.IO.File.WriteAllText(filePath, csvData);
        Debug.Log($"Native Plugins Performance Metrics written to \"{filePath}\"");

        // Set exit condition to be true
        exitFlag = true;
    }
}

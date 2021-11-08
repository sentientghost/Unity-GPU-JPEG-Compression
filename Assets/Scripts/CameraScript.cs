using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Rendering;

public class CameraScript : MonoBehaviour 
{
    [Header("Camera Settings")]
    [SerializeField] Camera cameraObject;
    [Header("Ball Settings")]
    public GameObject ballPrefab;
    GameObject ballObject;
    bool bufferFlag = true;
    bool exitFlag = false;
    bool testFlag = true;
    bool checkFlag = true;
    int cameraFrequency = 25;
    int bufferCount = 0;
    int bufferTime = 2; // in terms of seconds
    int frameCount;
    int[] resolutions;
    int[] qualities;
    float intervalTime;
    float timeElapsed;
    float bufferTimeElapsed;
    int[] imageSize = new int[2];
    int imageCount;
    LinearScript linearScript;
    CoroutinesScript coroutinesScript;

    int codeCount;
    int resolutionCount;
    int qualityCount;
    string buildMode;

    // sceneMetrics such that:
    // first bracket is code types (linear, coroutines)
    // second bracket is resolutions (144, 240, 360, 480, 720, 1080, 1440)
    // third bracket is jpeg qualities (75, 80, 85, 90, 95, 100)
    // fourth bracket is performance metrics (render, copy, encode, write)
    // fifth bracket is raw data for that performance metric (250 samples)
    float[,,,,] sceneMetrics = new float[2, 7, 6, 4, 250];

    void Awake() 
    {
        resolutions = new int[] {144, 240, 360, 480, 720, 1080, 1440};
        qualities = new int[] {75, 80, 85, 90, 95, 100};
        imageSize = CalculateImageSize(resolutions[0]);
    }

    void Start() 
    {
        if (Application.isEditor)
        {
            buildMode = "Editor";
        }
        else if (Application.isBatchMode)
        {
            buildMode = "Build-Headless";
        }
        else
        {
            buildMode = "Build-Windowed";
        }

        // Initialise Linear and Coroutines Scripts
        linearScript = gameObject.GetComponent<LinearScript> ();
        coroutinesScript = gameObject.GetComponent<CoroutinesScript> ();

        string objectName = SceneManager.GetActiveScene().name + "SceneBall";
        ballObject = GameObject.Find(objectName);
        
        frameCount = 0;
        imageCount = 0;

        codeCount = 0; 
        resolutionCount = 0; 
        qualityCount = 0; 

        intervalTime = Time.fixedTime + (1.0f/cameraFrequency);

        //pause the game physics
        Physics.autoSimulation = false;
    }

    void Update() 
    {
        timeElapsed += Time.deltaTime;

        if (SceneManager.GetActiveScene().buildIndex == 3 && exitFlag == true)
        {
            // conditional compilation
            #if UNITY_EDITOR
            EditorApplication.isPlaying = false;
            #else
            Application.Quit();
            #endif
        }
        else if (exitFlag == true)
        {
            // unloads current scene and loads the next scene in the build index
            SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex + 1, LoadSceneMode.Single);
        }
    }

    void FixedUpdate() 
    {
        if (Time.fixedTime >= intervalTime)
        {   
            if (bufferFlag == true)
            {
                if (bufferCount == (bufferTime * cameraFrequency))
                {
                    bufferCount = 0;
                    bufferFlag = false;

                    Physics.autoSimulation = true;

                    Debug.Log("Buffer Time Elasped: " + (timeElapsed - bufferTimeElapsed) + " s");
                }
                else if (bufferCount == 0)
                {
                    bufferTimeElapsed = timeElapsed;

                    Destroy(ballObject);
                    ballObject = Instantiate(ballPrefab);

                    Physics.autoSimulation = false;

                    bufferCount += 1;
                }
                else
                {
                    bufferCount += 1;
                }
            }
            else
            {
                if (testFlag)
                {
                    if (frameCount >= 250)
                    {
                        frameCount = 0;
                        imageCount += 1;

                        float progress = 100 * ((imageCount*250 + frameCount) / 21000f);

                        Debug.Log("Progress: " + progress.ToString("F2") + " %");
                        Debug.Log("Current Count Value: " + codeCount + " " + resolutionCount + " " + qualityCount);

                        IncrementCounts();
                    }
                    else if (!checkFlag)
                    {
                        frameCount -= 1;
                        checkFlag = SaveScreenJPG();
                        frameCount += 1;
                    }
                    else
                    {
                        checkFlag = SaveScreenJPG();
                        frameCount += 1;
                    }
                }
                else
                {
                    OutputMetrics();
                }
                
            }
            
            intervalTime = Time.fixedTime + (1.0f/cameraFrequency);
        }
    }

    void OnDisable()
    {
        Debug.Log("Total Time Elasped: " + timeElapsed + " s");
        Debug.Log("Total Images Saved: " + (imageCount*250 + frameCount));
        Debug.Log("Success!!! Test Completed");
    }

    int[] CalculateImageSize(int resolution)
    {
        float aspectRatio = 16.0f / 9.0f;
        int[] imageWH = new int[2] {(int)(resolution * aspectRatio), resolution};

        UnityEngine.Object.Destroy(cameraObject.targetTexture);
        cameraObject.targetTexture = new RenderTexture(imageWH[0], imageWH[1], 24);

        return imageWH;
    }

    void IncrementCounts()
    {
        if (codeCount == 1 && resolutionCount == 6 && qualityCount == 5)
        {
            testFlag = false;
            return;
        }

        qualityCount += 1;
        
        if (qualityCount >= 6)
        {
            qualityCount = 0;
            resolutionCount += 1;

            if (resolutionCount >= 7)
            {
                codeCount = 1;
                resolutionCount = 0;
            }
        }

        bufferFlag = true;
    }

    bool SaveScreenJPG ()
    {
        float[] imageTimes = new float[4];
        int[] imageWH = CalculateImageSize(resolutions[resolutionCount]);

        if (codeCount == 0)
        {
            imageTimes = linearScript.CallTakeImage(imageWH[0], imageWH[1], cameraObject, qualities[qualityCount], frameCount);
        }
        else if (codeCount == 1)
        {
            imageTimes = coroutinesScript.CallTakeImage(imageWH[0], imageWH[1], cameraObject, qualities[qualityCount], frameCount);
        }

        if (imageTimes[0] == 0 && imageTimes[1] == 0 && imageTimes[2] == 0 && imageTimes[3] == 0)
        {
            return false;
        }
        else
        {
            for (int metricCount = 0; metricCount <= 3; metricCount++)
            {
                sceneMetrics[codeCount, resolutionCount, qualityCount, metricCount, frameCount] = imageTimes[metricCount];
            }
            return true;
        }
    }

    string ProcessData()
    {
        StringBuilder csvData = new StringBuilder();

        // column headings
        string[] codeType = new string[2] {"linear", "coroutines"};
        string[] imageResolution = new string[7] {"144p", "240p", "360p", "480p", "720p", "1080p", "1440p"};
        string[] qualityLevel = new string[6] {"75", "80", "85", "90", "95", "100"};
        string[] metric = new string[4] {"render", "copy", "encode", "write"};
        
        for (int i = -1; i < 250; i++)
        {
            for(int j = 0; j < 2; j++)
            {
                for(int k = 0; k < 7; k++)
                {
                    for(int l = 0; l < 6; l++)
                    {
                        for(int m = 0; m < 4; m++)
                        {
                            if (i == -1)
                            {
                                string columnHeading = codeType[j] + "_" + imageResolution[k] + "_" + qualityLevel[l] + "_" + metric[m] + ",";
                                csvData.Append(columnHeading);
                            }
                            else
                            {
                                csvData.Append(sceneMetrics[j, k, l, m, i].ToString()).Append(",");
                            }
                        }
                    }
                }
            }
            csvData.Append("\n");
        } 

        return csvData.ToString();
    }

    string FilePath()
    {
        if (buildMode == "Editor")
        {
            return string.Format("{0}/../Metrics/Current Performance/{1}-Scene_{2}-Mode.csv", Application.dataPath, SceneManager.GetActiveScene().name, buildMode);
        }
        else
        {
            return string.Format("{0}/../../../Metrics/Current Performance/{1}-Scene_{2}-Mode.csv", Application.dataPath, SceneManager.GetActiveScene().name, buildMode);
        }    
    }

    void OutputMetrics()
    {
        string csvData = ProcessData();
        string filePath = FilePath();
        System.IO.File.WriteAllText(filePath, csvData);
        Debug.Log($"Current Performance Metrics written to \"{filePath}\"");
        exitFlag = true;
    }
}

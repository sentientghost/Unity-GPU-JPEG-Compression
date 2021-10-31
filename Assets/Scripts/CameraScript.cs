using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;


public class CameraScript : MonoBehaviour 
{
    [Header("Camera Settings")]
    [SerializeField] Camera cameraObject;
    GameObject ballObject;
    bool bufferFlag = true;
    bool exitFlag = false;
    bool testFlag = true;
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

    // staticScene such that:
    // first bracket is code types (linear, coroutines)
    // second bracket is resolutions (144, 240, 360, 480, 720, 1080, 1440)
    // third bracket is jpeg qualities (75, 80, 85, 90, 95, 100)
    // fourth bracket is performance metrics (render, copy, encode, write)
    float[,,,] staticScene = new float[2, 7, 6, 4];

    void Awake() 
    {
        resolutions = new int[] {144, 240, 360, 480, 720, 1080, 1440};
        qualities = new int[] {75, 80, 85, 90, 95, 100};
        imageSize = CalculateImageSize(resolutions[0]);
        cameraObject.targetTexture = new RenderTexture(imageSize[0], imageSize[1], 24);
    }

    void Start() 
    {
        // Initialise Linear and Coroutines Scripts
        linearScript = gameObject.GetComponent<LinearScript> ();
        coroutinesScript = gameObject.GetComponent<CoroutinesScript> ();

        ballObject = GameObject.Find("Ball");
        
        frameCount = 0;
        imageCount = 0;

        codeCount = 0; 
        resolutionCount = 0; 
        qualityCount = 0; 

        intervalTime = Time.fixedTime + (1.0f/cameraFrequency);
    }

    void Update() 
    {
        timeElapsed += Time.deltaTime;

        if (exitFlag == true)
        {
            EditorApplication.isPlaying = false;
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

                    Debug.Log("Buffer Time Elasped: " + (timeElapsed - bufferTimeElapsed) + " s");
                }
                else if (bufferCount == 0)
                {
                    bufferTimeElapsed = timeElapsed;

                    ballObject.transform.position = new Vector3(0, 2, 0);
                    ballObject.transform.rotation = Quaternion.Euler(0, 0, 0);
                    ballObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
                    ballObject.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;

                    bufferCount += 1;
                }
                else
                {
                    bufferCount += 1;
                }
            }
            else
            {

                if (frameCount >= 250)
                {
                    frameCount = 0;
                    imageCount += 1;

                    Debug.Log("Time Elasped: " + timeElapsed + " s");
                    Debug.Log("Total Images Saved: " + (imageCount*250 + frameCount));
                    Debug.Log("Counts: " + codeCount + " " + resolutionCount + " " + qualityCount);

                    IncrementCounts();

                    bufferFlag = true;
                }
                else
                {
                    frameCount += 1;
                }

                //SaveScreenJPG();

                if (!testFlag)
                {
                    exitFlag = true;
                }
            }
            
            intervalTime = Time.fixedTime + (1.0f/cameraFrequency);
        }
    }

    void OnDisable()
    {
        Debug.Log("Time Elasped: " + timeElapsed + " s");
        Debug.Log("Total Images Saved: " + (imageCount*250 + frameCount));
        Debug.Log("Raw Data stored to: -");
        Debug.Log("Test Completed");
    }

    int[] CalculateImageSize(int resolution)
    {
        float aspectRatio = 16.0f / 9.0f;
        int[] imageWH = new int[2] {resolution, (int)(resolution * aspectRatio)};
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
    }

    void SaveScreenJPG ()
    {
        float[] imageTimes = new float[4];
        int[] imageWH = CalculateImageSize(resolutions[resolutionCount]);

        if (codeCount == 0)
        {
            imageTimes = linearScript.CallTakeImage(imageWH[0], imageWH[1], cameraObject, qualities[qualityCount]);
        }
        else if (codeCount == 1)
        {
            imageTimes = coroutinesScript.CallTakeImage(imageWH[0], imageWH[1], cameraObject, qualities[qualityCount]);
        }

        for (int metricCount = 0; metricCount <= 3; metricCount++)
        {
            staticScene[codeCount, resolutionCount, qualityCount, metricCount] = imageTimes[metricCount];
        }
    }
}

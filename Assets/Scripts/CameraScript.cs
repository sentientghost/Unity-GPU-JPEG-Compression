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
    float[] imageTimes = new float[4];
    float[] times = new float[4];
    int imageCountIdeal = 0;
    int imageCountActual = 0;

    LinearScript linear;
    CoroutinesScript coroutines;
    JobsScript jobs;
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
    }

    void Start() 
    {
        //linear = gameObject.GetComponent<LinearScript> ();
        //coroutines = gameObject.GetComponent<CoroutinesScript> ();
        jobs = gameObject.GetComponent<JobsScript> ();

        intervalTime = Time.fixedTime + (1.0f/cameraFrequency);
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

    void FixedUpdate() 
    {
        if ((Time.fixedTime >= intervalTime) && (imageCountActual <= 244))
        {
            //imageTimes = linear.CallTakeImage(imageWidth, imageHeight, cameraObject, cameraQuality);
            //imageTimes = coroutines.StartTakeImage(imageWidth, imageHeight, cameraObject, cameraQuality);
            imageTimes = jobs.CallTakeImage(imageWidth, imageHeight, cameraObject, cameraQuality);
            times[0] += imageTimes[0];
            times[1] += imageTimes[1];
            times[2] += imageTimes[2];
            times[3] += imageTimes[3];
            imageCountActual += 1;
            intervalTime = Time.fixedTime + (1.0f/cameraFrequency);
        }
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

        // Calculate Average
        times[0] = times[0] / imageCountActual;
        times[1] = times[1] / imageCountActual;
        times[2] = times[2] / imageCountActual;
        times[3] = times[3] / imageCountActual;

        Debug.Log("Time Elasped: " + timeElapsed + " s");
        Debug.Log("Image Count: " + imageCountIdeal + " (ideal), " + imageCountActual + " (actual)");
        Debug.Log("Render: " + times[0] + " ms, Read/Copy: " + times[1] + " ms, Encode/Compress: " + times[2] + " ms, Write/Save: " + times[3] + " ms");
    }
}

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
    }

    void Start() 
    {
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

    void FixedUpdate() {
        if (Time.fixedTime >= intervalTime) {
            CallTakeImage();
            imageCountActual += 1;
            intervalTime = Time.fixedTime + (1.0f/cameraFrequency);
        }
    }

    void CallTakeImage()
    {
        Texture2D image = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        cameraObject.Render();
        RenderTexture.active = cameraObject.targetTexture;
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        byte[] bytes = image.EncodeToJPG(cameraQuality);
        string filename = ImageName();
        System.IO.File.WriteAllBytes(filename, bytes);
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
        
        Debug.Log("Time Elasped: " + timeElapsed);
        Debug.Log("Image Count (ideal):" + imageCountIdeal);
        Debug.Log("Image Count (actual):" + imageCountActual);
    }
}

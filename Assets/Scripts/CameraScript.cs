using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;


public class CameraScript : MonoBehaviour 
{
    [SerializeField] Camera cameraObject;
    [SerializeField] int cameraQuality = 75;
    [SerializeField] int cameraFrequency = 25;
    [SerializeField] float cameraTime = 10.0f;
    [SerializeField] int imageWidth = 1280;
    [SerializeField] int imageHeight = 720;

    float intervalTime;
    float timeElapsed;
    int imageCountIdeal = 0;
    int imageCountActual = 0;

    void Awake() 
    {
        if (cameraObject.targetTexture == null)
        {
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
        return string.Format("{0}/../Images/image_{1}x{2}_{3}.jpg", Application.dataPath, imageWidth, imageHeight, System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
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

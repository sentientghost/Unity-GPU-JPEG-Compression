using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Ensure always using Camera Script on a Camera Object
// [RequireComponent(typeof(Camera))]
public class CameraScript : MonoBehaviour 
{
    [SerializeField] Camera backCam;

    [SerializeField] int resWidth = 1280;
    [SerializeField] int resHeight = 720;

    [SerializeField] int frequency = 25;
    float intervalTime;

    // void Awake() 
    // {
    //     //backCam = GetComponent<Camera>();

    //     if (backCam.targetTexture == null)
    //     {
    //         backCam.targetTexture = new RenderTexture(resWidth, resHeight, 24);
    //     }
    //     else
    //     {
    //         resWidth = backCam.targetTexture.width;
    //         resHeight = backCam.targetTexture.height;
    //     }

    //     backCam.gameObject.SetActive(false);
    // }

    void Start() {
        backCam.targetTexture = new RenderTexture(resWidth, resHeight, 24);
        intervalTime = Time.fixedTime + (1.0f/frequency);
    }

    void FixedUpdate() {
        if (Time.fixedTime >= intervalTime) {
            CallTakeImage();
            intervalTime = Time.fixedTime + (1.0f/frequency);
        }
    }

    void CallTakeImage()
    {
        Texture2D image = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
        backCam.Render();
        RenderTexture.active = backCam.targetTexture;
        image.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
        byte[] bytes = image.EncodeToPNG();
        string filename = ImageName();
        System.IO.File.WriteAllBytes(filename, bytes);
        Debug.Log("Image Taken!");
    }

    string ImageName()
    {
        return string.Format("{0}/../Images/image_{1}x{2}_{3}.png", Application.dataPath, resWidth, resHeight, System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }
}

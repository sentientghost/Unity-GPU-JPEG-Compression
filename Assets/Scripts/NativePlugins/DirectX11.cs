using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;


public class DirectX11 : MonoBehaviour
{
    //the name of the DLL you want to load stuff from
    private const string pluginName = "DirectX11";


    // Access C++ function from C# script
    [DllImport(pluginName)]
    private static extern int SimpleReturnFunc();


    // Use this for initialization
    void Start () {
        Debug.Log(SimpleReturnFunc());
    }
}

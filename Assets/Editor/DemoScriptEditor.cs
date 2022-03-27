using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(DemoScript))]
class DemoScriptEditor : Editor 
{
    // Override inspector from Demo Script
    public override void OnInspectorGUI()
    {
        // Draw the default inspector from the Demo Script
        DrawDefaultInspector();
        
        // Add functionality to select output directory from Windows file explorer
        DemoScript demoScript = (DemoScript) target;

        if (GUILayout.Button("Select Output Directory"))
        {
            string outputDirectory = EditorUtility.OpenFolderPanel("Select Output Directory", "", "");
            demoScript.outputDirectory = outputDirectory;

            if (demoScript.verbose)
            {
                Debug.Log("Output Directory: " + outputDirectory);
            }
        }
    }
}

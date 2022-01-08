![project_banner](RepoImages/ProjectBanner.png)

# Unity-On-GPU-JPEG-Compression

Develop a method for compressing GPU textures using the JPEG specification, in Unity, and then evaluate its performance.

# Project Objectives

1. Understand how JPEG compression works to reduce the size of raw image data
2. Understand the Unity Rendering pipeline
3. Investigate different methods currently used for on-GPU JPEG compression
4. Investigate the current performance of image generation and saving in Unity.
5. Select a suitable approach and implement a prototype.
6. Measure performance (both Unity performance and render quality) of the system after your custom
on-GPU compressor is used and note any improvements. Determine if there is a performance
improvement

# Repo Structure
```
GitHub Landing Page: Explains repository structure and contains a single Unity project for all possible solutions
├── Assets: Contains images for banners and logos
│   ├── Lighting: Contains the lighting maps for the scenes
│   ├── Materials: Contains different materials for the objects in the scenes
│   ├── Prefabs: Contains prefab objects to spawn into the scenes
│   ├── Scenes: Contains the scenes used to test the pipeline
│   └── Scripts: Contains C# scripts used in the project
│       ├── CameraScript: Main script that acts as a camera controller
│       ├── LinearScript: Take image using a linear coding approach
│       └── CoroutinesScript: Take image using a coroutines coding approach 
|
├── Packages: Contains packages installed from the Unity Package Manager
|
├── RepoImages: Contains images for banners and logos
|
└── README.md
```

# Requirements

Below is a list of hardware and software you will need to get started:

- Hardware
    - NVIDIA Graphics Card
    - At least 8GB Ram
- Software
    - Windows 10 (note that other Windows versions have NOT been tested)
    - Unity 2020.3 LTS (note that other Unity versions have NOT been tested)

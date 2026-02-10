// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "YoloV8Detector.h"


extern "C" {

    // Helper to cast void* to YoloV8Detector*
    inline YoloV8Detector* GetDetector(void* detector) {
        return static_cast<YoloV8Detector*>(detector);
    }

    /// <summary>
    /// Creates a new instance of the YoloV8Detector.
    /// </summary>
    /// <returns>A pointer (handle) to the new detector instance.</returns>
    __declspec(dllexport) void* CreateDetector() {
        return new YoloV8Detector();
    }

    /// <summary>
    /// Destroys the specified detector instance and frees memory.
    /// </summary>
    /// <param name="detector">The detector handle to destroy.</param>
    __declspec(dllexport) void DestroyDetector(void* detector) {
        if (detector) {
            delete GetDetector(detector);
        }
    }

    /// <summary>
    /// Loads the model for the specified detector instance.
    /// </summary>
    /// <param name="detector">The detector handle.</param>
    /// <param name="modelPath">Path to the model file.</param>
    /// <param name="deviceNum">Device ID (0 for CPU, 1 for CUDA).</param>
    /// <returns>1 on success, -1 on failure.</returns>
    __declspec(dllexport) int LoadModel(void* detector, char* modelPath, int deviceNum) {
        if (!detector) return -1;
        return GetDetector(detector)->LoadModel(modelPath, deviceNum);
    }

    /// <summary>
    /// Sets the thresholds for the specified detector instance.
    /// </summary>
    /// <param name="detector">The detector handle.</param>
    /// <param name="score_thresh">Confidence score threshold.</param>
    /// <param name="iou_thresh">IoU threshold.</param>
    /// <param name="seg_thresh">Segmentation threshold.</param>
    __declspec(dllexport) void SetThreshold(void* detector, float score_thresh, float iou_thresh, float seg_thresh) {
        if (detector) {
            GetDetector(detector)->SetThreshold(score_thresh, iou_thresh, seg_thresh);
        }
    }

    /// <summary>
    /// Performs inference on an image path using the specified detector.
    /// </summary>
    __declspec(dllexport) int PerformImagePathInference(void* detector, char* imgPath, int net_height, int net_width) {
        if (!detector) return -1;
        return GetDetector(detector)->Detect(imgPath, net_height, net_width);
    }

    /// <summary>
    /// Performs inference on raw image data using the specified detector.
    /// </summary>
    __declspec(dllexport) int PerformFrameInference(void* detector, uchar* inputData, int net_height, int net_width) {
        if (!detector) return -1;
        
        // We assume inputData is a BGR image buffer of size net_width x net_height.
        // The C# wrapper ensures resizing before calling this.
        return GetDetector(detector)->Detect(inputData, net_height, net_width, net_height, net_width);
    }

    /// <summary>
    /// Populates the array of detected objects.
    /// </summary>
    __declspec(dllexport) void PopulateYoloObjectsArray(void* detector, YoloObject* objects, int org_height, int org_width) {
        if (detector) {
            GetDetector(detector)->PopulateObjects(objects, org_height, org_width);
        }
    }
}

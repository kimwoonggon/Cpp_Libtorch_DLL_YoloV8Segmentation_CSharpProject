// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include "YoloV8Detector.h"


extern "C" {

    // [CHANGE]
    // Before: Global functions manipulating global state.
    //         Example: __declspec(dllexport) int LoadModel(...) { ... modifies global 'network' ... }
    //
    // After:  "Opaque Pointer" (Handle) pattern.
    //         We explicitly create an instance (`CreateDetector`) and pass it to every function (`detector`).
    //         This clearly identifies WHICH detector we are communicating with.

    // Helper to cast void* to YoloV8Detector*
    inline YoloV8Detector* GetDetector(void* detector) {
        return static_cast<YoloV8Detector*>(detector);
    }

    // [CHANGE] NEW Function
    // Purpose: Allocates a new YoloV8Detector instance and returns its address.
    __declspec(dllexport) void* CreateDetector() {
        return new YoloV8Detector();
    }

    // [CHANGE] NEW Function
    // Purpose: Cleanly deletes the instance. 
    // Before: `FreeAllocatedMemory()` only cleared vectors but didn't destroy any "Object" because there wasn't one.
    __declspec(dllexport) void DestroyDetector(void* detector) {
        if (detector) {
            delete GetDetector(detector);
        }
    }

    // [CHANGE]
    // Before: LoadModel(char* modelPath, int deviceNum)
    // After:  LoadModel(void* detector, char* modelPath, int deviceNum)
    //         We must specify WHICH detector instance should load the model.
    __declspec(dllexport) int LoadModel(void* detector, char* modelPath, int deviceNum) {
        if (!detector) return -1;
        return GetDetector(detector)->LoadModel(modelPath, deviceNum);
    }

    __declspec(dllexport) void SetThreshold(void* detector, float score_thresh, float iou_thresh, float seg_thresh) {
        if (detector) {
            GetDetector(detector)->SetThreshold(score_thresh, iou_thresh, seg_thresh);
        }
    }

    __declspec(dllexport) int PerformImagePathInference(void* detector, char* imgPath, int net_height, int net_width) {
        if (!detector) return -1;
        return GetDetector(detector)->Detect(imgPath, net_height, net_width);
    }

    __declspec(dllexport) int PerformFrameInference(void* detector, uchar* inputData, int net_height, int net_width) {
        if (!detector) return -1;
        // Passed inputData is assumed to be resized to net_height x net_width 
        // because that's what the C# wrapper does before calling this.
        return GetDetector(detector)->Detect(inputData, net_height, net_width, net_height, net_width);
    }

    __declspec(dllexport) void PopulateYoloObjectsArray(void* detector, YoloObject* objects, int org_height, int org_width) {
        if (detector) {
            GetDetector(detector)->PopulateObjects(objects, org_height, org_width);
        }
    }

    // Legacy functions or other initialization if required
    BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
    {
        switch (ul_reason_for_call)
        {
        case DLL_PROCESS_ATTACH:
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
        case DLL_PROCESS_DETACH:
            break;
        }
        return TRUE;
    }
}
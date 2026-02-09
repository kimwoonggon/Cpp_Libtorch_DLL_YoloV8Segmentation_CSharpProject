#pragma once
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>

// Structure to represent a detected object (Internal/External shared)
// [CHANGE] This file is NEW.
// Before: All state (network, device_type, buffers) was declared as GLOBAL variables in dllmain.cpp.
//         Example: extern "C" { torch::jit::Module network; ... }
//
// After:  All state is encapsulated in this YoloV8Detector class.
//         This allows:
//         1. Multiple instances (e.g., loading different models).
//         2. Thread safety (each instance has its own state).
//         3. Automatic resource cleanup via Destructor.

struct YoloObject {
    float left;
    float top;
    float right;
    float bottom;
    float score;
    float classID;
    uchar* seg_map; // Pointer to segmentation map data
};

class YoloV8Detector {
public:
    YoloV8Detector();
    ~YoloV8Detector();

    // [CHANGE]
    // Before: global logic inside LoadModel(char* modelPath, ...)
    // After:  method acting on 'this' instance.
    int LoadModel(const std::string& modelPath, int deviceNum);
    
    // Set thresholds
    void SetThreshold(float score_thresh, float iou_thresh, float seg_thresh);

    // Perform inference on an image path
    int Detect(const std::string& imgPath, int net_height, int net_width);

    // Perform inference on raw image data
    int Detect(uchar* inputData, int rows, int cols, int net_height, int net_width);

    // Get detected objects and populate the array
    void PopulateObjects(YoloObject* objects, int org_height, int org_width);

private:
    // Internal helper for NMS
    void non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh);

    torch::jit::Module network;
    torch::DeviceType device_type;

    std::string _modelName;
    float _score_thresh = 0.5f;
    float _iou_thresh = 0.5f;
    float _seg_thresh = 0.5f;

    // Inference State
    std::vector<torch::Tensor> dets_vec;
    torch::Tensor predLoc;
    torch::Tensor seg_pred;
    
    // Visualization / Segmentation State
    cv::Mat total_seg_map;
    int real_net_width;
    int real_net_height;

    // Pre-defined colors
    std::vector<cv::Vec3b> colors;
    
    // Helper to initialize colors
    void InitializeColors();
};

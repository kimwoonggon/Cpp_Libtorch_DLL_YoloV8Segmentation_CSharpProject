#pragma once
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>

/// <summary>
/// Structure to represent a single detected object.
/// This structure is shared between the C++ DLL and the C# application.
/// </summary>
struct YoloObject {
    float left;         ///< Bounding box left coordinate
    float top;          ///< Bounding box top coordinate
    float right;        ///< Bounding box right coordinate
    float bottom;       ///< Bounding box bottom coordinate
    float score;        ///< Confidence score of the detection
    float classID;      ///< Class ID of the detected object
    uchar* seg_map;     ///< Pointer to segmentation map data (if applicable)
};

/// <summary>
/// A class that encapsulates the YOLOv8 segmentation logic using LibTorch.
/// </summary>
class YoloV8Detector {
public:
    /// <summary>
    /// Constructor for YoloV8Detector.
    /// Initializes the detector instance.
    /// </summary>
    YoloV8Detector();

    /// <summary>
    /// Destructor for YoloV8Detector.
    /// Cleans up resources.
    /// </summary>
    ~YoloV8Detector();

    /// <summary>
    /// Loads the YOLOv8 TorchScript model.
    /// </summary>
    /// <param name="modelPath">The file path to the TorchScript (.torchscript) model.</param>
    /// <param name="deviceNum">The device to use for inference (0: CPU, 1: CUDA).</param>
    /// <returns>Returns 1 if the model loads successfully, 0 otherwise.</returns>
    int LoadModel(const std::string& modelPath, int deviceNum);
    
    /// <summary>
    /// Sets the thresholds for detection and segmentation.
    /// </summary>
    /// <param name="score_thresh">Confidence score threshold for detections.</param>
    /// <param name="iou_thresh">IoU threshold for Non-Maximum Suppression (NMS).</param>
    /// <param name="seg_thresh">Threshold for segmentation mask binaryzation.</param>
    void SetThreshold(float score_thresh, float iou_thresh, float seg_thresh);

    /// <summary>
    /// Performs inference on an image file.
    /// </summary>
    /// <param name="imgPath">The path to the input image.</param>
    /// <param name="net_height">The height of the network input (e.g., 640).</param>
    /// <param name="net_width">The width of the network input (e.g., 640).</param>
    /// <returns>Returns the number of objects detected.</returns>
    int Detect(const std::string& imgPath, int net_height, int net_width);

    /// <summary>
    /// Performs inference on raw image data.
    /// </summary>
    /// <param name="inputData">Pointer to the raw image data (BGR format).</param>
    /// <param name="rows">The height of the input image.</param>
    /// <param name="cols">The width of the input image.</param>
    /// <param name="net_height">The height of the network input.</param>
    /// <param name="net_width">The width of the network input.</param>
    /// <returns>Returns the number of objects detected.</returns>
    int Detect(uchar* inputData, int rows, int cols, int net_height, int net_width);

    /// <summary>
    /// Populates an array of YoloObject structures with the detection results.
    /// </summary>
    /// <param name="objects">Pointer to the array of YoloObject structures to populate.</param>
    /// <param name="org_height">The original height of the image (for scaling coordinates).</param>
    /// <param name="org_width">The original width of the image (for scaling coordinates).</param>
    void PopulateObjects(YoloObject* objects, int org_height, int org_width);

private:
    /// <summary>
    /// Performs Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.
    /// </summary>
    /// <param name="preds">The raw predictions from the model.</param>
    /// <param name="score_thresh">The confidence score threshold.</param>
    /// <param name="iou_thresh">The IoU threshold.</param>
    void non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh);

    torch::jit::Module network;       ///< The loaded LibTorch model module
    torch::DeviceType device_type;    ///< The device type (CPU or CUDA)

    std::string _modelName;           ///< The name/path of the loaded model
    float _score_thresh = 0.5f;       ///< Default score threshold
    float _iou_thresh = 0.5f;         ///< Default IoU threshold
    float _seg_thresh = 0.5f;         ///< Default segmentation threshold

    // Inference State
    std::vector<torch::Tensor> dets_vec; ///< Vector to store detection tensors
    torch::Tensor predLoc;            ///< Tensor for predicted locations
    torch::Tensor seg_pred;           ///< Tensor for segmentation predictions
    
    // Visualization / Segmentation State
    cv::Mat total_seg_map;            ///< The combined segmentation map
    int real_net_width;               ///< Network width used in inference
    int real_net_height;              ///< Network height used in inference

    // Pre-defined colors
    std::vector<cv::Vec3b> colors;    ///< List of colors for class visualization
    
    /// <summary>
    /// Initializes the colors vector with random colors.
    /// </summary>
    void InitializeColors();
};

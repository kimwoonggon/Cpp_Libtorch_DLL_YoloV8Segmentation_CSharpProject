// dllmain.cpp : Defines the entry point for the DLL application.
// dllmain.cpp
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#include "pch.h"

namespace fs = std::filesystem;

extern "C" {
// The Yolov8 segmentation model loaded from a file, used for inference.
torch::jit::Module network;
// The type of device on which the network will run (e.g., CPU, CUDA).
torch::DeviceType device_type;

// The actual dimensions of the network input.
int real_net_width;
int real_net_height;
// Original image dimensions before resizing for the network.
int _org_height;
int _org_width;

// Thresholds for filtering out predictions with low confidence.
float _score_thresh = 0.5f;  // Minimum score to consider a detection valid.
float _iou_thresh = 0.5f;    // IOU threshold for non-maximum suppression.
float _seg_thresh = 0.5f;    // Threshold for segmentation mask confidence.

// A vector to hold detection results (bounding boxes, scores, etc.).
std::vector<torch::Tensor> dets_vec;

// Name or identifier for the model
std::string _modelName;

// Tensors holding the predicted locations and segmentation predictions.
torch::Tensor predLoc;   // Predicted bounding box locations.
torch::Tensor seg_pred;  // Predicted segmentation masks.

// An OpenCV matrix to hold the aggregated segmentation map.
cv::Mat total_seg_map;

// Struct to represent a detected object.
struct YoloObject {
  float left;      // Left coordinate of the bounding box.
  float top;       // Top coordinate of the bounding box.
  float right;     // Right coordinate of the bounding box.
  float bottom;    // Bottom coordinate of the bounding box.
  float score;     // Confidence score of the class detection.
  float classID;   // ID of the detected class.
  uchar* seg_map;  // Pointer to the segmentation map for this object.
};

// A predefined set of colors for visualizing detection and segmentation
// results.
std::vector<cv::Vec3b> colors = {
    cv::Vec3b(0, 0, 255),      // Red
    cv::Vec3b(0, 255, 0),      // Green
    cv::Vec3b(255, 0, 0),      // Blue
    cv::Vec3b(255, 255, 0),    // Cyan
    cv::Vec3b(255, 0, 255),    // Magenta
    cv::Vec3b(0, 255, 255),    // Yellow
    cv::Vec3b(128, 0, 0),      // Dark Red
    cv::Vec3b(0, 128, 0),      // Dark Green
    cv::Vec3b(0, 0, 128),      // Dark Blue
    cv::Vec3b(128, 128, 0),    // Olive
    cv::Vec3b(128, 0, 128),    // Purple
    cv::Vec3b(0, 128, 128),    // Teal
    cv::Vec3b(192, 192, 192),  // Silver
    cv::Vec3b(128, 128, 128),  // Gray
    cv::Vec3b(64, 0, 0),       // Maroon
    cv::Vec3b(0, 64, 0),       // Dark green
    cv::Vec3b(0, 0, 64),       // Navy
    cv::Vec3b(64, 64, 0),      // Dark Olive
    cv::Vec3b(64, 0, 64),      // Indigo
    cv::Vec3b(0, 64, 64),      // Dark Cyan
    cv::Vec3b(192, 192, 0),    // Mustard
    cv::Vec3b(192, 0, 192),    // Pink
    cv::Vec3b(0, 192, 192),    // Sky Blue
    cv::Vec3b(64, 192, 0),     // Lime Green
    cv::Vec3b(192, 64, 0),     // Orange
    cv::Vec3b(0, 192, 64),     // Sea Green
    cv::Vec3b(64, 0, 192),     // Royal Blue
    cv::Vec3b(192, 0, 64),     // Deep Pink
    cv::Vec3b(0, 64, 192),     // Cerulean
    cv::Vec3b(64, 192, 192),   // Turquoise
    cv::Vec3b(192, 64, 192),   // Orchid
    cv::Vec3b(192, 192, 64),   // Sand
    cv::Vec3b(128, 64, 64),    // Rosy Brown
    cv::Vec3b(64, 128, 64),    // Pale Green
    cv::Vec3b(64, 64, 128),    // Slate Blue
    cv::Vec3b(128, 128, 64),   // Khaki
    cv::Vec3b(128, 64, 128),   // Plum
    cv::Vec3b(64, 128, 128),   // Cadet Blue
    cv::Vec3b(140, 70, 20),    // Saddle Brown
    cv::Vec3b(0, 140, 140),    // Dark Turquoise
    cv::Vec3b(0, 0, 255),      // Red
    cv::Vec3b(0, 255, 0),      // Green
    cv::Vec3b(255, 0, 0),      // Blue
    cv::Vec3b(255, 255, 0),    // Cyan
    cv::Vec3b(255, 0, 255),    // Magenta
    cv::Vec3b(0, 255, 255),    // Yellow
    cv::Vec3b(128, 0, 0),      // Dark Red
    cv::Vec3b(0, 128, 0),      // Dark Green
    cv::Vec3b(0, 0, 128),      // Dark Blue
    cv::Vec3b(128, 128, 0),    // Olive
    cv::Vec3b(128, 0, 128),    // Purple
    cv::Vec3b(0, 128, 128),    // Teal
    cv::Vec3b(192, 192, 192),  // Silver
    cv::Vec3b(128, 128, 128),  // Gray
    cv::Vec3b(64, 0, 0),       // Maroon
    cv::Vec3b(0, 64, 0),       // Dark green
    cv::Vec3b(0, 0, 64),       // Navy
    cv::Vec3b(64, 64, 0),      // Dark Olive
    cv::Vec3b(64, 0, 64),      // Indigo
    cv::Vec3b(0, 64, 64),      // Dark Cyan
    cv::Vec3b(192, 192, 0),    // Mustard
    cv::Vec3b(192, 0, 192),    // Pink
    cv::Vec3b(0, 192, 192),    // Sky Blue
    cv::Vec3b(64, 192, 0),     // Lime Green
    cv::Vec3b(192, 64, 0),     // Orange
    cv::Vec3b(0, 192, 64),     // Sea Green
    cv::Vec3b(64, 0, 192),     // Royal Blue
    cv::Vec3b(192, 0, 64),     // Deep Pink
    cv::Vec3b(0, 64, 192),     // Cerulean
    cv::Vec3b(64, 192, 192),   // Turquoise
    cv::Vec3b(192, 64, 192),   // Orchid
    cv::Vec3b(192, 192, 64),   // Sand
    cv::Vec3b(128, 64, 64),    // Rosy Brown
    cv::Vec3b(64, 128, 64),    // Pale Green
    cv::Vec3b(64, 64, 128),    // Slate Blue
    cv::Vec3b(128, 128, 64),   // Khaki
    cv::Vec3b(128, 64, 128),   // Plum
    cv::Vec3b(64, 128, 128),   // Cadet Blue
    cv::Vec3b(140, 70, 20),    // Saddle Brown
    cv::Vec3b(0, 140, 140),    // Dark Turquoise
};

// Exports the function for external use in DLL format.
__declspec(dllexport) void SetDevice(int deviceNum) {
  // Set the device for model computations. 0 for CPU, any other number for GPU
  // if available.
  if (deviceNum == 0) {
    device_type = torch::kCPU;  // Use CPU for computations.
  } else {
    device_type =
        torch::kCUDA;  // Use GPU (CUDA) for computations if available.
  }
}

// Exports the function for external use. Loads the model and sets the
// computation device.
__declspec(dllexport) int LoadModel(char* modelPath, int deviceNum) {
  int return_val = 1;  // Return value indicating success (1) or failure (-1).
  try {
    // Check the model name to ensure it's supported (e.g., yolov8).
    if (strstr(modelPath, "yolov8")) {
      _modelName = "yolov8";  // Set the model name if it matches.
    } else {
      return_val = -1;  // If the model name doesn't match, return an error.
      return return_val;
    }
    // Set the computation device based on the input parameter.
    if (deviceNum == 0) {
      device_type = torch::kCPU;  // Use CPU for computations.
    } else {
      // If GPU is requested, check if CUDA is available.
      if (torch::cuda::is_available()) {
        device_type =
            torch::kCUDA;  // Use GPU (CUDA) for computations if available.
      } else {
        device_type = torch::kCPU;  // Fallback to CPU if CUDA is not available.
      }
    }
    // Load the model from the specified path onto the selected device.
    network = torch::jit::load(modelPath, device_type);
    network.eval();  // Set the network to evaluation mode (disables dropout, etc..)
                     
    std::cout << "device type : " << device_type
              << std::endl;  // Debug: print the selected device type.
  } catch (const c10::Error& e) {
    std::cout << "Model reading failed .. "
              << std::endl;  // Handle errors in model loading.
    return -1;
  }
  return return_val;  // Return success status.
}

// Exports the function to set global thresholds for detection and segmentation.
__declspec(dllexport) void SetThreshold(float score_thresh,
                                        float iou_thresh,
                                        float seg_thresh) {
  _score_thresh =
      score_thresh;  // Minimum confidence score to consider a detection valid.
  _iou_thresh = iou_thresh;  // IOU threshold for non-maximum suppression
                             // (filtering overlapping boxes)..
  _seg_thresh = seg_thresh;  // Threshold for segmentation mask confidence.
}

// Applies Non-Maximum Suppression to filter overlapping detections.
void non_max_suppression(
    torch::Tensor preds,         // Model predictions [1,8400,116]
    float score_thresh = 0.5,    // Default score threshold.
    float iou_thresh = 0.5) {    // Default IOU threshold for NMS.
  dets_vec.clear();              // Clear previous detections.
  auto device = preds.device();  // Get the device of the predictions tensor.
  for (size_t i = 0; i < preds.sizes()[0]; ++i) {
    torch::Tensor pred =
        preds.select(0, i).to(device);  // Process each prediction.
    // If using a YOLOv8 model, filter detections based on score threshold.
    if (_modelName == "yolov8") {
      torch::Tensor scores =
          std::get<0>(torch::max(pred.slice(1, 4, 84), 1));  // Get scores.
      // Filter out low-score predictions.
      pred = torch::index_select(
          pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
    } else {
      throw std::runtime_error("Model name is not valid");
    }
    if (pred.sizes()[0] == 0)  // Skip if no predictions left after filtering.
      continue;
    // Convert bounding box format from center x, center y, width, height (cx,
    // cy, w, h) to top-left and bottom-right corners (x1, y1, x2, y2).
    pred.select(1, 0) =
        pred.select(1, 0) - pred.select(1, 2) / 2;  // Calculate x1
    pred.select(1, 1) =
        pred.select(1, 1) - pred.select(1, 3) / 2;              // Calculate y1
    pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);  // Calculate x2
    pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);  // Calculate y2

    // Identify the maximum confidence score for each prediction and its
    // corresponding class.
    auto max_tuple = torch::max(pred.slice(1, 4, 84), 1);
    pred.select(1, 4) = std::get<0>(max_tuple);  // Set max confidence score
    predLoc = std::get<1>(max_tuple).to(pred.device());  // Store class id

    torch::Tensor dets;
    // Combine bounding box coordinates with confidence scores and class ids
    // into a single tensor.
    dets = torch::cat({pred.slice(1, 0, 5), pred.slice(1, 84, 116)}, 1);

    // Prepare tensors to keep track of indices of detections to retain.
    torch::Tensor keep = torch::empty({dets.sizes()[0]}, dets.options());
    torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) *
                          (dets.select(1, 2) - dets.select(1, 0));

    // Sort detections by confidence score in descending order.
    auto indexes_tuple = torch::sort(dets.select(1, 4), 0,
                                     1);  // 0: first order, 1: decending order

    torch::Tensor v = std::get<0>(indexes_tuple);
    torch::Tensor indexes = std::get<1>(indexes_tuple);

    int count = 0;  // Counter for detections to keep.

    // Loop over detections and apply non-maximum suppression.
    while (indexes.sizes()[0] > 0) {
      // Always keep the detection with the highest current score.
      keep[count++] = (indexes[0].item().toInt());
      // Compute the pairwise overlap between the highest scoring detection and
      // all others. Preallocate tensors to hold the computed overlaps.
      torch::Tensor lefts =
          torch::empty(indexes.sizes()[0] - 1, indexes.options());
      torch::Tensor tops =
          torch::empty(indexes.sizes()[0] - 1, indexes.options());
      torch::Tensor rights =
          torch::empty(indexes.sizes()[0] - 1, indexes.options());
      torch::Tensor bottoms =
          torch::empty(indexes.sizes()[0] - 1, indexes.options());
      torch::Tensor widths =
          torch::empty(indexes.sizes()[0] - 1, indexes.options());
      torch::Tensor heights =
          torch::empty(indexes.sizes()[0] - 1, indexes.options());

      // Loop over each detection remaining after the one with the highest
      // score.
      for (size_t i = 0; i < indexes.sizes()[0] - 1; ++i) {
        // Compute the coordinates of the intersection rectangle.
        lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(),
                            dets[indexes[i + 1]][0].item().toFloat());
        tops[i] = std::max(dets[indexes[0]][1].item().toFloat(),
                           dets[indexes[i + 1]][1].item().toFloat());
        rights[i] = std::min(dets[indexes[0]][2].item().toFloat(),
                             dets[indexes[i + 1]][2].item().toFloat());
        bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(),
                              dets[indexes[i + 1]][3].item().toFloat());
        widths[i] = std::max(
            float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
        heights[i] = std::max(
            float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
      }

      // Compute the intersection over union (IoU) for each pair.
      torch::Tensor overlaps = widths * heights;
      torch::Tensor ious =
          overlaps / (areas.select(0, indexes[0].item().toInt()) +
                      torch::index_select(
                          areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) -
                      overlaps);
      auto nonzero_indices = torch::nonzero(ious <= iou_thresh);
      torch::Tensor kk = torch::nonzero(ious <= iou_thresh).select(1, 0) + 1;
      // Filter out detections with IoU above the threshold, as they overlap too
      // much with the highest scoring box.
      indexes = torch::index_select(
          indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
    }

    // Convert the 'keep' tensor to 64-bit integer type. This is necessary for
    // indexing operations that follow.
    keep = keep.toType(torch::kInt64);

    // Select the detections that have been marked for keeping.
    dets_vec.emplace_back(std::move(
        torch::index_select(dets, 0, keep.slice(0, 0, count)).to(torch::kCPU)));

    // Similarly, select the locations (predLoc) corresponding to the kept
    // detections.
    predLoc = torch::index_select(predLoc, 0, keep.slice(0, 0, count))
                  .to(torch::kCPU);
  }
}


// Exports function for DLL, performs inference on an single image file.
__declspec(dllexport) int PerformImagePathInference(char* imgPath,
                                           int net_height,
                                           int net_width) {
  real_net_height =
      net_height;  // Set the global variable for network input height.
  real_net_width =
      net_width;  // Set the global variable for network input width.

  // Read the input image from the provided path.
  cv::Mat input_img = cv::imread(imgPath);
  if (input_img.empty()) {  // Check if the image was successfully read.
    std::cout << "Could not read the image: " << imgPath << std::endl;
    return -1;  // Return error if image read fails.
  }

  // Resize the image to match the network input dimensions.
  cv::resize(input_img, input_img, cv::Size(net_width, net_height));

  // Convert the color space from BGR to RGB, which is expected by most models.
  cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);

  // Normalize the image by converting its pixel values to float and scaling
  // down by 255.
  input_img.convertTo(input_img, CV_32FC3, 1.0f / 255.0f);

  // Convert the OpenCV image to a Torch tensor.
  torch::Tensor imgTensor =
      torch::from_blob(input_img.data, {net_width, net_height, 3})
          .to(device_type);

  // Permute dimensions to match the model's expected input [C, H, W] format
  imgTensor = imgTensor.permute({2, 0, 1}).contiguous();
  // Add a batch dimension.
  imgTensor = imgTensor.unsqueeze(0);

  // Prepare the tensor for model input.
  std::vector<torch::jit::IValue> inputs;
  imgTensor.to(device_type);

  inputs.push_back(std::move(imgTensor));
  try {
    // Enable inference mode for efficiency.
    torch::InferenceMode guard(true);
    // Forward pass: run the model with the input tensor.
    torch::jit::IValue output = network.forward(inputs);
    // Extract predictions.
    auto preds = output.toTuple()->elements()[0].toTensor();

    // Model-specific adjustments (e.g., YOLOv8 requires transposing).
    if (_modelName == "yolov8") {
      preds = preds.transpose(1, 2).contiguous();
    }
    // Extract segmentation predictions if present.
    seg_pred = output.toTuple()->elements()[1].toTensor();

    // Apply non-maximum suppression to filter out overlapping detections.
    non_max_suppression(preds, _score_thresh, _iou_thresh);

    // Return the number of detections.
    int return_size = dets_vec[0].sizes()[0];
    return return_size;

  } catch (const c10::Error& e) {
    std::cerr << e.what() << std::endl;
    return -1;  // Return error on exception.
  }
}

// Performs inference on image data provided in memory, useful for video or
// webcam streams.
__declspec(dllexport) int PerformFrameInference(uchar* inputData,
                                                int net_height,
                                                int net_width) {
  real_net_height = net_height;  // Global network input height.
  real_net_width = net_width;    // Global network input width.

  // Create an OpenCV Mat from the raw input data.
  cv::Mat input_img2 = cv::Mat(net_height, net_width, CV_8UC3, inputData);
  // Convert BGR to RGB.
  cv::cvtColor(input_img2, input_img2, cv::COLOR_BGR2RGB);
  // Normalize the image by converting its pixel values to float and scaling
  // down by 255.
  input_img2.convertTo(input_img2, CV_32FC3, 1.0f / 255.0f);
  // Convert the OpenCV Mat to a Torch tensor.
  torch::Tensor imgTensor =
      torch::from_blob(input_img2.data, {net_width, net_height, 3})
          .to(device_type);
  // Adjust tensor dimensions to match model's input [C, H, W] format.
  imgTensor = imgTensor.permute({2, 0, 1}).contiguous();
  // Add a batch dimension.
  imgTensor = imgTensor.unsqueeze(0);
  imgTensor.to(device_type);

  // Prepare for model input.
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(std::move(imgTensor));

  try {
    // Enable inference mode.
    torch::InferenceMode guard(true);
    // Forward pass with the provided data.
    torch::jit::IValue output = network.forward(inputs);

    // Process the model's output (similar to the previous function).
    auto preds = output.toTuple()->elements()[0].toTensor();
    if (_modelName == "yolov8") {
      preds = preds.transpose(1, 2).contiguous();
    }
    seg_pred = output.toTuple()->elements()[1].toTensor();
    non_max_suppression(preds, _score_thresh, _iou_thresh);

    // Check if there are any detections.
    if (dets_vec.size() == 0) {
      return 0;  // No detections found.
    } else {
      // Return the number of detections.
      torch::Tensor det = dets_vec[0];
      int size = det.sizes()[0];
      return size;  // Return the number of detections.
    }

  } catch (const c10::Error& e) {
    std::cerr << e.what() << std::endl;
    return -1;  // Return error on exception.
  }
}

// Exports the function for DLL use, intended to organize detected objects and
// segmentation results.
__declspec(dllexport) void PopulateYoloObjectsArray(YoloObject* objects,
                                                int org_height,
                                                int org_width) {
  // Early return if no detections were made.
  if (dets_vec.size() == 0) {
    return;
  }
  // Access the first tensor in the detections vector.
  torch::Tensor det = dets_vec[0];
  // Get the number of detections.
  int size = det.sizes()[0];

  // Initialize an empty segmentation map with the original image dimensions.
  total_seg_map = cv::Mat(org_height, org_width, CV_8UC3, cv::Scalar(0, 0, 0));

  // Iterate over each detection.
  for (int i = 0; i < size; i++) {
    // Scale bounding box coordinates from the network size to the original
    // image size.
    float left = det[i][0].item().toFloat() * org_width /
                 real_net_width;  // Ensure left is within image bounds.
    left = std::max(0.0f, left);  // Ensure left is within image bounds.
    float top = det[i][1].item().toFloat() * org_height / real_net_height;
    top = std::max(top, 0.0f);  // Ensure top is within image bounds.
    float right = det[i][2].item().toFloat() * org_width / real_net_width;
    right = std::min(
        right,
        (float)(org_width - 1));  // Ensure right does not exceed image width.
    float bottom = det[i][3].item().toFloat() * org_height / real_net_height;
    bottom = std::min(
        bottom, (float)(org_height -
                        1));  // Ensure bottom does not exceed image height.
    float score =
        det[i][4].item().toFloat();  // Get the detection confidence score.

    // Assign detection properties to the objects array.
    objects[i].left = left;
    objects[i].top = top;
    objects[i].right = right;
    objects[i].bottom = bottom;
    objects[i].score = score;

    int classID;             // Variable to store class ID
    torch::Tensor seg_rois;  // Tensor to hold segmentation regions of interest.

    // Check if the model is yolov8.
    if (_modelName == "yolov8") {
      classID = predLoc[i].item().toInt();  // Extract class ID.
      seg_rois =
          det[i].slice(0, 5, det[i].sizes()[0]);  // Extract segmentation ROI.
      objects[i].classID = classID;
    } else {
      throw std::runtime_error("Model name is not valid");
    }

    // Prepare segmentation mask.
    seg_rois = seg_rois.view({1, 32});
    seg_pred = seg_pred.to(torch::kCPU);
    seg_pred = seg_pred.view({1, 32, -1});
    auto final_seg = torch::matmul(seg_rois, seg_pred).view({1, 160, 160});
    final_seg =
        final_seg.sigmoid();  // Apply sigmoid to get mask probabilities.
    final_seg = ((final_seg > _seg_thresh) * 255)
                    .clamp(0, 255)
                    .to(torch::kCPU)
                    .to(torch::kU8);
    // Convert probabilities to binary mask.
    cv::Mat seg_map(160, 160, CV_8UC1,
                    final_seg.data_ptr());  // Resize to original image size.
    cv::Mat seg_map2;
    cv::resize(seg_map, seg_map2, cv::Size(org_width, org_height),
               cv::INTER_LINEAR);
    cv::Mat seg_map_color;
    cv::cvtColor(seg_map2, seg_map_color,
                 cv::COLOR_GRAY2BGR);  // Convert grayscale to BGR.

    // Colorize the segmentation map.
    for (int y = 0; y < seg_map_color.rows; y++) {
      for (int x = 0; x < seg_map_color.cols; x++) {
        if (seg_map_color.at<cv::Vec3b>(y, x)[0] > 0) {
          seg_map_color.at<cv::Vec3b>(y, x) =
              colors[classID];  // Apply class-specific color.
        } else
          seg_map_color.at<cv::Vec3b>(y, x) =
              cv::Vec3b(0, 0, 0);  // Set background to black.
      }
    }
    // Combine current object's segmentation with the total segmentation map.
    cv::bitwise_or(total_seg_map, seg_map_color, total_seg_map);

    // Optional: Save segmentation map for debugging.
    /*std::string path =
        "seg_map_" + std::to_string(i) + ".png";
    cv::imwrite(path, total_seg_map);*/
  }
  // Segmap is shared among all objects
  for (int i = 0; i < size; i++) {
    objects[i].seg_map = total_seg_map.data;
  }
}

// Exports the function for DLL, designed to free up resources used during
// inference.
__declspec(dllexport) void FreeAllocatedMemory() {
  dets_vec.clear();  // Clear the detections vector to free up memory.

  // If the model was loaded onto a CUDA (GPU) device, move it back to CPU to
  // free GPU resources.
  if (device_type == torch::kCUDA) {
    network.to(torch::kCPU);
  }

  // Reset the network variable, effectively releasing the loaded model from
  // memory.
  network = torch::jit::Module();

  // Release the OpenCV matrix holding the segmentation map, freeing its memory.
  total_seg_map.release();
}





}
#include "pch.h"
#include "YoloV8Detector.h"
#include <filesystem>
#include <iostream>

YoloV8Detector::YoloV8Detector() {
    InitializeColors();
    device_type = torch::kCPU;
}

YoloV8Detector::~YoloV8Detector() {
    // Torch modules and tensors usually handle their own memory, 
    // but we might want to ensure OpenCV mats are released if they invoke specific allocators.
    total_seg_map.release();
}

void YoloV8Detector::InitializeColors() {
    // Populate colors (same as original code)
    colors = {
        cv::Vec3b(0, 0, 255),      cv::Vec3b(0, 255, 0),      cv::Vec3b(255, 0, 0),
        cv::Vec3b(255, 255, 0),    cv::Vec3b(255, 0, 255),    cv::Vec3b(0, 255, 255),
        cv::Vec3b(128, 0, 0),      cv::Vec3b(0, 128, 0),      cv::Vec3b(0, 0, 128),
        cv::Vec3b(128, 128, 0),    cv::Vec3b(128, 0, 128),    cv::Vec3b(0, 128, 128),
        cv::Vec3b(192, 192, 192),  cv::Vec3b(128, 128, 128),  cv::Vec3b(64, 0, 0),
        cv::Vec3b(0, 64, 0),       cv::Vec3b(0, 0, 64),       cv::Vec3b(64, 64, 0),
        cv::Vec3b(64, 0, 64),      cv::Vec3b(0, 64, 64),      cv::Vec3b(192, 192, 0),
        cv::Vec3b(192, 0, 192),    cv::Vec3b(0, 192, 192),    cv::Vec3b(64, 192, 0),
        cv::Vec3b(192, 64, 0),     cv::Vec3b(0, 192, 64),     cv::Vec3b(64, 0, 192),
        cv::Vec3b(192, 0, 64),     cv::Vec3b(0, 64, 192),     cv::Vec3b(64, 192, 192),
        cv::Vec3b(192, 64, 192),   cv::Vec3b(192, 192, 64),   cv::Vec3b(128, 64, 64),
        cv::Vec3b(64, 128, 64),    cv::Vec3b(64, 64, 128),    cv::Vec3b(128, 128, 64),
        cv::Vec3b(128, 64, 128),   cv::Vec3b(64, 128, 128),   cv::Vec3b(140, 70, 20),
        cv::Vec3b(0, 140, 140),    cv::Vec3b(0, 0, 255),      cv::Vec3b(0, 255, 0),
        cv::Vec3b(255, 0, 0),      cv::Vec3b(255, 255, 0),    cv::Vec3b(255, 0, 255),
        cv::Vec3b(0, 255, 255),    cv::Vec3b(128, 0, 0),      cv::Vec3b(0, 128, 0),
        cv::Vec3b(0, 0, 128),      cv::Vec3b(128, 128, 0),    cv::Vec3b(128, 0, 128),
        cv::Vec3b(0, 128, 128),    cv::Vec3b(192, 192, 192),  cv::Vec3b(128, 128, 128),
        cv::Vec3b(64, 0, 0),       cv::Vec3b(0, 64, 0),       cv::Vec3b(0, 0, 64),
        cv::Vec3b(64, 64, 0),      cv::Vec3b(64, 0, 64),      cv::Vec3b(0, 64, 64),
        cv::Vec3b(192, 192, 0),    cv::Vec3b(192, 0, 192),    cv::Vec3b(0, 192, 192),
        cv::Vec3b(64, 192, 0),     cv::Vec3b(192, 64, 0),     cv::Vec3b(0, 192, 64),
        cv::Vec3b(64, 0, 192),     cv::Vec3b(192, 0, 64),     cv::Vec3b(0, 64, 192),
        cv::Vec3b(64, 192, 192),   cv::Vec3b(192, 64, 192),   cv::Vec3b(192, 192, 64),
        cv::Vec3b(128, 64, 64),    cv::Vec3b(64, 128, 64),    cv::Vec3b(64, 64, 128),
        cv::Vec3b(128, 128, 64),   cv::Vec3b(128, 64, 128),   cv::Vec3b(64, 128, 128),
        cv::Vec3b(140, 70, 20),    cv::Vec3b(0, 140, 140)
    };
}

int YoloV8Detector::LoadModel(const std::string& modelPath, int deviceNum) {
    if (modelPath.find("yolov8") != std::string::npos) {
        _modelName = "yolov8";
    } else {
        return -1;
    }

    if (deviceNum == 0) {
        device_type = torch::kCPU;
    } else {
        if (torch::cuda::is_available()) {
            device_type = torch::kCUDA;
        } else {
            device_type = torch::kCPU;
        }
    }

    try {
        network = torch::jit::load(modelPath, device_type);
        network.eval();
        std::cout << "Device type: " << device_type << std::endl;
        return 1;
    } catch (const c10::Error& e) {
        std::cerr << "Model loading failed: " << e.what() << std::endl;
        return -1;
    }
}

void YoloV8Detector::SetThreshold(float score_thresh, float iou_thresh, float seg_thresh) {
    _score_thresh = score_thresh;
    _iou_thresh = iou_thresh;
    _seg_thresh = seg_thresh;
}

void YoloV8Detector::non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh) {
    dets_vec.clear();
    auto device = preds.device();
    
    for (size_t i = 0; i < preds.sizes()[0]; ++i) {
        torch::Tensor pred = preds.select(0, i).to(device);
        
        if (_modelName == "yolov8") {
            torch::Tensor scores = std::get<0>(torch::max(pred.slice(1, 4, 84), 1));
            pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
        } else {
            throw std::runtime_error("Model name is not valid");
        }

        if (pred.sizes()[0] == 0) continue;

        pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
        pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
        pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
        pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

        auto max_tuple = torch::max(pred.slice(1, 4, 84), 1);
        pred.select(1, 4) = std::get<0>(max_tuple);
        predLoc = std::get<1>(max_tuple).to(pred.device());

        torch::Tensor dets = torch::cat({pred.slice(1, 0, 5), pred.slice(1, 84, 116)}, 1);

        torch::Tensor keep = torch::empty({dets.sizes()[0]}, dets.options());
        torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));

        auto indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
        torch::Tensor indexes = std::get<1>(indexes_tuple);

        int count = 0;
        while (indexes.sizes()[0] > 0) {
            keep[count++] = (indexes[0].item().toInt());
            
            torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1, indexes.options());
            torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1, indexes.options());
            torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1, indexes.options());
            torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1, indexes.options());
            torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1, indexes.options());
            torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1, indexes.options());

            for (size_t k = 0; k < indexes.sizes()[0] - 1; ++k) {
                lefts[k] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[k + 1]][0].item().toFloat());
                tops[k] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[k + 1]][1].item().toFloat());
                rights[k] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[k + 1]][2].item().toFloat());
                bottoms[k] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[k + 1]][3].item().toFloat());
                widths[k] = std::max(float(0), rights[k].item().toFloat() - lefts[k].item().toFloat());
                heights[k] = std::max(float(0), bottoms[k].item().toFloat() - tops[k].item().toFloat());
            }

            torch::Tensor overlaps = widths * heights;
            torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + 
                                           torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
            
            indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
        }

        keep = keep.toType(torch::kInt64);
        dets_vec.emplace_back(std::move(torch::index_select(dets, 0, keep.slice(0, 0, count)).to(torch::kCPU)));
        predLoc = torch::index_select(predLoc, 0, keep.slice(0, 0, count)).to(torch::kCPU);
    }
}


// [CHANGE]
// Before: Global function `PerformImagePathInference` directly accessed global `network` and `device_type`.
// After:  Member function `Detect` uses `this->network` and `this->device_type`.
int YoloV8Detector::Detect(const std::string& imgPath, int net_height, int net_width) {
    real_net_height = net_height;
    real_net_width = net_width;

    cv::Mat input_img = cv::imread(imgPath);
    if (input_img.empty()) {
        std::cout << "Could not read the image: " << imgPath << std::endl;
        return -1;
    }

    // [CHANGE] Preprocessing logic remains largely the same, but operates on local/member variables.
    cv::resize(input_img, input_img, cv::Size(net_width, net_height));
    cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
    input_img.convertTo(input_img, CV_32FC3, 1.0f / 255.0f);

    torch::Tensor imgTensor = torch::from_blob(input_img.data, {net_width, net_height, 3}).to(device_type);
    imgTensor = imgTensor.permute({2, 0, 1}).contiguous();
    imgTensor = imgTensor.unsqueeze(0);

    // [CHANGE] Added RAII guard for InferenceMode (cleaner than manual handling if any)
    torch::InferenceMode guard(true);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(std::move(imgTensor));

    try {
        // [CHANGE] Accessing member `network` instead of global `network`.
        torch::jit::IValue output = network.forward(inputs);
        auto preds = output.toTuple()->elements()[0].toTensor();

        if (_modelName == "yolov8") {
            preds = preds.transpose(1, 2).contiguous();
        }
        seg_pred = output.toTuple()->elements()[1].toTensor();

        // [CHANGE] Call internal private helper instead of global function
        non_max_suppression(preds, _score_thresh, _iou_thresh);

        if (dets_vec.empty()) return 0;
        return dets_vec[0].sizes()[0];

    } catch (const c10::Error& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}

int YoloV8Detector::Detect(uchar* inputData, int rows, int cols, int net_height, int net_width) {
    real_net_height = net_height;
    real_net_width = net_width;

    cv::Mat input_img2 = cv::Mat(rows, cols, CV_8UC3, inputData);
    cv::cvtColor(input_img2, input_img2, cv::COLOR_BGR2RGB);
    
    // We need to resize here as well if the input size is not match network size
    // In original code, it seemed to assume inputData was already resized OR 
    // it created a mat of net_height/width directly. 
    // Let's stick to original logic: "cv::Mat(net_height, net_width, ...)" 
    // implies the input buffer matches net dimensions? 
    // Actually the original code signature was `PerformFrameInference(uchar* inputData, int net_height, int net_width)`.
    // And it did `cv::Mat(net_height, net_width, CV_8UC3, inputData)`. 
    // This implies the inputData IS ALREADY resized to net_height x net_width. 
    // BUT usually `rows` and `cols` are variable. 
    // I will assume for this overload, the C# side passes the properly sized buffer or we handle resizing?
    // Let's correct this. The original code creates mat of size `net_height, net_width`. 
    // This means the C# side MUST pass a resized image or the pointer interprets it as such.
    // However, C# code `ReturnFramePerformance` passes `img.DataPointer` where `img` is resized to `net_width, net_height`.
    // So `rows` and `cols` passed to this function SHOULD be net_height/net_width.
    
    cv::Mat resized_img; 
    // If input is not already float/normalized:
    input_img2.convertTo(resized_img, CV_32FC3, 1.0f / 255.0f);

    torch::Tensor imgTensor = torch::from_blob(resized_img.data, {net_width, net_height, 3}).to(device_type);
    imgTensor = imgTensor.permute({2, 0, 1}).contiguous();
    imgTensor = imgTensor.unsqueeze(0);

    torch::InferenceMode guard(true);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(std::move(imgTensor));

    try {
        torch::jit::IValue output = network.forward(inputs);
        auto preds = output.toTuple()->elements()[0].toTensor();
        
        if (_modelName == "yolov8") {
            preds = preds.transpose(1, 2).contiguous();
        }
        seg_pred = output.toTuple()->elements()[1].toTensor();
        
        non_max_suppression(preds, _score_thresh, _iou_thresh);

        if (dets_vec.empty()) return 0;
        return dets_vec[0].sizes()[0];
    } catch (const c10::Error& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}

void YoloV8Detector::PopulateObjects(YoloObject* objects, int org_height, int org_width) {
    if (dets_vec.empty()) return;

    torch::Tensor det = dets_vec[0];
    int size = det.sizes()[0];

    // Re-initialize segmentation map
    total_seg_map = cv::Mat(org_height, org_width, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < size; i++) {
        float left = std::max(0.0f, det[i][0].item().toFloat() * org_width / real_net_width);
        float top = std::max(0.0f, det[i][1].item().toFloat() * org_height / real_net_height);
        float right = std::min((float)(org_width - 1), det[i][2].item().toFloat() * org_width / real_net_width);
        float bottom = std::min((float)(org_height - 1), det[i][3].item().toFloat() * org_height / real_net_height);
        float score = det[i][4].item().toFloat();

        objects[i].left = left;
        objects[i].top = top;
        objects[i].right = right;
        objects[i].bottom = bottom;
        objects[i].score = score;
        
        int classID = predLoc[i].item().toInt();
        objects[i].classID = (float)classID;

        torch::Tensor seg_rois = det[i].slice(0, 5, det[i].sizes()[0]); 

        // Process segmentation
        seg_rois = seg_rois.view({1, 32});
        // Ensure seg_pred is on CPU
        torch::Tensor seg_pred_cpu = seg_pred.to(torch::kCPU);
        seg_pred_cpu = seg_pred_cpu.view({1, 32, -1});
        
        auto final_seg = torch::matmul(seg_rois, seg_pred_cpu).view({1, 160, 160});
        final_seg = final_seg.sigmoid();
        final_seg = ((final_seg > _seg_thresh) * 255).clamp(0, 255).to(torch::kCPU).to(torch::kU8);

        cv::Mat seg_map(160, 160, CV_8UC1, final_seg.data_ptr());
        cv::Mat seg_map2;
        cv::resize(seg_map, seg_map2, cv::Size(org_width, org_height), cv::INTER_LINEAR);
        cv::Mat seg_map_color;
        cv::cvtColor(seg_map2, seg_map_color, cv::COLOR_GRAY2BGR);

        // Colorize
        auto color = (classID < colors.size()) ? colors[classID] : cv::Vec3b(255, 255, 255);
        cv::Scalar colorScalar(color[0], color[1], color[2]);
        
        cv::Mat mask = (seg_map2 > 0);
        seg_map_color.setTo(colorScalar, mask);
        seg_map_color.setTo(cv::Scalar(0,0,0), ~mask);

        cv::bitwise_or(total_seg_map, seg_map_color, total_seg_map);
    }

    // Assign shared seg map pointer
    for (int i = 0; i < size; i++) {
        objects[i].seg_map = total_seg_map.data;
    }
}

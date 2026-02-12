# GPU-Accelerated YOLOv8 Segmentation in C++ with LibTorch DLLs and C# Integration

This repository provides an end-to-end guide and necessary tools to perform GPU-accelerated YOLOv8 segmentation using C++ DLLs, integrated with C# via LibTorch and CUDA. It encompasses converting PyTorch YOLOv8 weights to TorchScript and ONNX, setting up C++ DLL projects with LibTorch and CUDA, and using these DLLs from C# for efficient inference.

> [!IMPORTANT]
> **Note regarding the Medium Article:**  
> The code in this repository (`master`) has been significantly improved for **safety and robustness** (memory management, thread safety) compared to the original version described in the Medium article.  
> If you are looking for the exact code matching the tutorial, please refer to **[Release v1.0](https://github.com/kimwoonggon/Cpp_Libtorch_DLL_YoloV8Segmentation_CSharpProject/releases/tag/v1.0)**.

## Recent Updates (Refactoring & Documentation)

The project has undergone significant refactoring to improve stability, memory management, and ease of use:

- **Class-based C++ Architecture**: The C++ DLL now uses a `YoloV8Detector` class to encapsulate the model state, replacing previous global variables. This supports multiple instances and better resource management.
- **Opaque Pointer API**: The C-API now uses an opaque pointer (Handle) pattern, ensuring clean separation between the C# wrapper and C++ internals.
- **Safe C# Wrapper**: A new `YoloDetector` class in C# wraps the DLL calls and implements `IDisposable`. This ensures that unmanaged C++ memory is automatically and safely released when the object is disposed or used in a `using` block.
- **Comprehensive Documentation**:
  - **C++**: Doxygen-style comments in `YoloV8Detector.h` and explanatory comments in `.cpp` files.
  - **C#**: XML documentation comments for Intellisense support in Visual Studio.

## Features

- **YOLOv8_Libtorch_Conversion.ipynb**: Notebook for converting YOLOv8 PyTorch weights to TorchScript and ONNX formats.
- **YoloV8DLLProject.sln**: Visual Studio 2022 solution file for the C++ DLL project setup with LibTorch and CUDA.
- **YoloV8DLLProject**: C++ DLL source code featuring the `YoloV8Detector` class for GPU-accelerated inference.
- **YoloCSharpInference**: C# project using the `YoloDetector` wrapper for easy and safe inference.

## Prerequisites

Before starting, ensure you have the following installed and properly configured on your system:
- CUDA and cuDNN (compatible versions with your LibTorch and PyTorch setup)
- LibTorch (PyTorch C++ distributions of GPU)
- .NET framework 4.7 +
- Visual Studio 2022 or later

Refer to the complete guide here for detailed setup instructions: [C++ DLL & C# with CUDA Libtorch: YOLOv8 Segmentation Guide](https://medium.com/@psopen11/complete-guide-to-gpu-accelerated-yolov8-segmentation-in-c-via-libtorch-c-dlls-a0e3e6029d82)

## Getting Started

1. **Clone the Repository**:
```
git clone https://github.com/kimwoonggon/Cpp_Libtorch_DLL_YoloV8Segmentation_CSharpProject
```
2. **Environment Setup**: Follow the detailed guide to set up your environment for CUDA, cuDNN, LibTorch, and C#. The guide provides step-by-step instructions for installation and configuration to ensure compatibility and performance.

3. **Convert YOLOv8 Weights**: Use the `YOLOv8_Libtorch_Conversion.ipynb` notebook to convert the YOLOv8 model weights into TorchScript and ONNX formats for use with LibTorch.

4. **Build the C++ DLL Project**: Open `YoloV8DLLProject.sln` in Visual Studio 2022, configure it for your system's CUDA and LibTorch setup, and build the project to generate the necessary DLLs.

5. **Run C# Inference**: With the DLLs built, you can now use the `YoloCSharpInference` project to load these DLLs and perform inference in C#.

## Usage

### C# Inference Example

With the new `YoloDetector` wrapper, inference is simple and safe:

```csharp
using YoloCShaprInference;

// Initialize detector (automatically disposed at end of block)
using (var detector = new YoloDetector()) 
{
    // Load model (0=CPU, 1=CUDA)
    if (detector.LoadModel("yolov8s-seg.torchscript", 1)) 
    {
        detector.SetThreshold(0.5f, 0.45f, 0.5f);
        
        // Run inference
        int count = detector.Detect("image.jpg", 640, 640);
        
        // Get results
        var objects = detector.GetDetectedObjects(count, orgHeight, orgWidth);
        
        // Process results...
    }
}
```

Detailed usage instructions for each component of the project are provided within their respective directories. This includes how to convert models, build DLLs, and perform inference using the C# project.

## Contributing

Contributions are welcome! If you have improvements or bug fixes, please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

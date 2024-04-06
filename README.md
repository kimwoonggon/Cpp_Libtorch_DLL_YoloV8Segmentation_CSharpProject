# GPU-Accelerated YOLOv8 Segmentation in C++ with LibTorch DLLs and C# Integration

This repository provides an end-to-end guide and necessary tools to perform GPU-accelerated YOLOv8 segmentation using C++ DLLs, integrated with C# via LibTorch and CUDA. It encompasses converting PyTorch YOLOv8 weights to TorchScript and ONNX, setting up C++ DLL projects with LibTorch and CUDA, and using these DLLs from C# for efficient inference.

## Features

- **YOLOv8_Libtorch_Conversion.ipynb**: Notebook for converting YOLOv8 PyTorch weights to TorchScript and ONNX formats.
- **YoloV8DLLProject.sln**: Visual Studio 2022 solution file for the C++ DLL project setup with LibTorch and CUDA.
- **YoloV8DLLProject**: Directory containing the C++ DLL source code for YOLOv8 segmentation with GPU acceleration.
- **YoloCSharpInference**: C# project for loading the C++ DLLs and performing inference.

## Prerequisites

Before starting, ensure you have the following installed and properly configured on your system:
- CUDA and cuDNN (compatible versions with your LibTorch and PyTorch setup)
- LibTorch (PyTorch C++ distributions)
- .NET framework or .NET Core for C# integration
- Visual Studio 2022 or later

Refer to the complete guide here for detailed setup instructions: [Complete Guide to GPU-Accelerated YOLOv8 Segmentation in C++ via LibTorch C++ DLLs](https://medium.com/@psopen11/complete-guide-to-gpu-accelerated-yolov8-segmentation-in-c-via-libtorch-c-dlls-a0e3e6029d82)

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

Detailed usage instructions for each component of the project are provided within their respective directories. This includes how to convert models, build DLLs, and perform inference using the C# project.

## Contributing

Contributions are welcome! If you have improvements or bug fixes, please feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

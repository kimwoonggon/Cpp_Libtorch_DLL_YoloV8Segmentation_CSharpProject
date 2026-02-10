using System;
using System.Runtime.InteropServices;
using System.IO;
using OpenCvSharp;

namespace YoloCShaprInference
{
    [StructLayout(LayoutKind.Sequential)]
    public struct YoloObject
    {
        public float left;
        public float top;
        public float right;
        public float bottom;
        public float score;
        public float classID;
        public IntPtr seg_map; // Generic pointer, used internally by C++
    }

    /// <summary>
    /// A wrapper class for the YOLOv8 C++ DLL.
    /// Manages the full lifecycle of the detector instance and provides safe methods for inference.
    /// </summary>
    public class YoloDetector : IDisposable
    {
        private IntPtr _detectorHandle; // Holds the pointer to the C++ 'YoloV8Detector' object
        private const string DllName = "YoloV8DLLProject";
        private bool _disposed = false;

        // DLL Imports
        // [CHANGE] Notice the first argument is now 'IntPtr detector'. 
        // We pass the instance handle to every function.
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr CreateDetector();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void DestroyDetector(IntPtr detector);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int LoadModel(IntPtr detector, string modelPath, int deviceNum);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void SetThreshold(IntPtr detector, float score_thresh, float iou_thresh, float seg_thresh);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int PerformImagePathInference(IntPtr detector, string imgPath, int net_height, int net_width);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int PerformFrameInference(IntPtr detector, IntPtr inputData, int net_height, int net_width);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void PopulateYoloObjectsArray(IntPtr detector, IntPtr objects, int org_height, int org_width);

        /// <summary>
        /// Initializes a new instance of the <see cref="YoloDetector"/> class.
        /// Creates a corresponding C++ detector instance.
        /// </summary>
        /// <exception cref="Exception">Thrown if the native instance cannot be created.</exception>
        public YoloDetector()
        {
            _detectorHandle = CreateDetector();
            if (_detectorHandle == IntPtr.Zero)
            {
                throw new Exception("Failed to create YoloV8Detector instance.");
            }
        }

        /// <summary>
        /// Loads the YOLOv8 model from the specified path.
        /// </summary>
        /// <param name="modelPath">The absolute path to the .torchscript model file.</param>
        /// <param name="deviceNum">The device to use for inference. 0 for CPU, 1 for CUDA (default).</param>
        /// <returns>True if the model was loaded successfully; otherwise, false.</returns>
        public bool LoadModel(string modelPath, int deviceNum = 1) // Default to CUDA
        {
            if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedException(nameof(YoloDetector));
            int result = LoadModel(_detectorHandle, modelPath, deviceNum);
            return result == 1;
        }

        /// <summary>
        /// Sets the internal thresholds for the detector.
        /// </summary>
        /// <param name="score">Confidence score threshold.</param>
        /// <param name="iou">IoU threshold for NMS.</param>
        /// <param name="seg">Segmentation mask threshold.</param>
        public void SetThreshold(float score, float iou, float seg)
        {
            if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedException(nameof(YoloDetector));
            SetThreshold(_detectorHandle, score, iou, seg);
        }

        /// <summary>
        /// Performs inference on an image file.
        /// </summary>
        /// <param name="imgPath">Path to the image file.</param>
        /// <param name="netHeight">Network input height.</param>
        /// <param name="netWidth">Network input width.</param>
        /// <returns>The number of objects detected.</returns>
        public int Detect(string imgPath, int netHeight, int netWidth)
        {
            if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedException(nameof(YoloDetector));
            return PerformImagePathInference(_detectorHandle, imgPath, netHeight, netWidth);
        }

        /// <summary>
        /// Performs inference on an in-memory image (OpenCvSharp Mat).
        /// </summary>
        /// <param name="image">The input image matrix.</param>
        /// <param name="netHeight">Network input height.</param>
        /// <param name="netWidth">Network input width.</param>
        /// <returns>The number of objects detected.</returns>
        public int Detect(Mat image, int netHeight, int netWidth)
        {
             if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedException(nameof(YoloDetector));
             // PerformFrameInference expects the image data to be properly resized to netHeight x netWidth
             // The C++ side reads it as netHeight x netWidth image.
             return PerformFrameInference(_detectorHandle, (IntPtr)image.Data, netHeight, netWidth);
        }

        /// <summary>
        /// Retrieves the detailed information for detected objects.
        /// </summary>
        /// <param name="numObjects">The number of objects returned by the Detect method.</param>
        /// <param name="orgHeight">Original image height for scaling coordinates.</param>
        /// <param name="orgWidth">Original image width for scaling coordinates.</param>
        /// <returns>An array of <see cref="YoloObject"/> containing detection details.</returns>
        public YoloObject[] GetDetectedObjects(int numObjects, int orgHeight, int orgWidth)
        {
            if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedException(nameof(YoloDetector));
            if (numObjects <= 0) return new YoloObject[0];

            YoloObject[] objects = new YoloObject[numObjects];
            
            // Pin the array to get a pointer
            GCHandle handle = GCHandle.Alloc(objects, GCHandleType.Pinned);
            try
            {
                IntPtr ptr = handle.AddrOfPinnedObject();
                PopulateYoloObjectsArray(_detectorHandle, ptr, orgHeight, orgWidth);
            }
            finally
            {
                handle.Free();
            }

            return objects;
        }

        /// <summary>
        /// Releases all resources used by the <see cref="YoloDetector"/>.
        /// Use this to ensure C++ memory is freed.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_detectorHandle != IntPtr.Zero)
                {
                    DestroyDetector(_detectorHandle);
                    _detectorHandle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~YoloDetector()
        {
            Dispose(false);
        }
    }
}

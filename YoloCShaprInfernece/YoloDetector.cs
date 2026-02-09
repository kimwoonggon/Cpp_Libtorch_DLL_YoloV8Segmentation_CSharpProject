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

    public class YoloDetector : IDisposable
    {
        private IntPtr _detectorHandle;
        private const string DllName = "YoloV8DLLProject";
        private bool _disposed = false;

        // DLL Imports
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

        public YoloDetector()
        {
            _detectorHandle = CreateDetector();
            if (_detectorHandle == IntPtr.Zero)
            {
                throw new Exception("Failed to create YoloV8Detector instance.");
            }
        }

        public bool LoadModel(string modelPath, int deviceNum = 1) // Default to CUDA
        {
            if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedExceptionAttribute(nameof(YoloDetector));
            int result = LoadModel(_detectorHandle, modelPath, deviceNum);
            return result == 1;
        }

        public void SetThreshold(float score, float iou, float seg)
        {
            if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedExceptionAttribute(nameof(YoloDetector));
            SetThreshold(_detectorHandle, score, iou, seg);
        }

        public int Detect(string imgPath, int netHeight, int netWidth)
        {
            if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedExceptionAttribute(nameof(YoloDetector));
            return PerformImagePathInference(_detectorHandle, imgPath, netHeight, netWidth);
        }

        public int Detect(Mat image, int netHeight, int netWidth)
        {
             if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedExceptionAttribute(nameof(YoloDetector));
             // PerformFrameInference expects the image data to be properly resized to netHeight x netWidth
             // The C++ side reads it as netHeight x netWidth image.
             return PerformFrameInference(_detectorHandle, image.DataPointer, netHeight, netWidth);
        }

        public YoloObject[] GetDetectedObjects(int numObjects, int orgHeight, int orgWidth)
        {
            if (_detectorHandle == IntPtr.Zero) throw new ObjectDisposedExceptionAttribute(nameof(YoloDetector));
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

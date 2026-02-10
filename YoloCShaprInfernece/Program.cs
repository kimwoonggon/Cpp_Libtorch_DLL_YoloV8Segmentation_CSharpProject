using System;
using System.Diagnostics;
using System.IO;
using OpenCvSharp;

namespace YoloCShaprInference
{
    internal class Program
    {
        public static string modelPath = "yolov8s-seg.torchscript";
        public static int net_height = 640;
        public static int net_width = 640;

        // Device selection: 0 for CPU, 1 for CUDA
        private static int deviceNum = 1; 

        static void tryFrameInference(string videoPath)
        {
            VideoCapture capture;
            if (!string.IsNullOrEmpty(videoPath) && System.IO.File.Exists(videoPath))
            {
                capture = new VideoCapture(videoPath);
            }
            else
            {
                capture = new VideoCapture(0);
            }

            if (!capture.IsOpened())
            {
                Console.WriteLine("Failed to open video source.");
                return;
            }

            // [CHANGE]
            // Before: `SetDevice(deviceNum); LoadModel(modelPath, deviceNum);` (Global calls)
            // After:  `using (var detector = new YoloDetector()) { ... }`
            //         This creates the instance, uses it, and AUTOMATICALLY destroys it (calling Dispose) when the block ends.
            //         This is much safer than manually calling `FreeAllocatedMemory()`.
            using (var detector = new YoloDetector())
            {
                if (!detector.LoadModel(modelPath, deviceNum))
                {
                    Console.WriteLine("Model not loaded");
                    return;
                }
                Console.WriteLine("Model Loaded.");

                detector.SetThreshold(0.3f, 0.3f, 0.3f);
                Console.WriteLine("Thresholds set.");

                var window = new Window("capture");
                var frame = new Mat();
                var img = new Mat();
                Stopwatch stopwatch = new Stopwatch();

                while (true)
                {
                    capture.Read(frame);
                    if (frame.Empty())
                    {
                        Console.WriteLine("End of stream or error.");
                        break;
                    }

                    stopwatch.Restart();

                    // Resize for inference
                    Cv2.Resize(frame, img, new Size(net_width, net_height));

                    // Perform Inference
                    // [CHANGE]
                    // Before: `PerformFrameInference(..., net_height, net_width)`
                    // After:  `detector.Detect(..., net_height, net_width)`
                    int numObjects = detector.Detect(img, net_height, net_width);

                    if (numObjects > 0)
                    {
                         // Get Results
                         var objects = detector.GetDetectedObjects(numObjects, frame.Height, frame.Width);
                         
                         // Draw Results
                         DrawResults(frame, objects);
                    }

                    stopwatch.Stop();
                    double fps = 1000.0 / stopwatch.ElapsedMilliseconds;
                    Cv2.PutText(frame, $"FPS: {fps:0.0}", new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.Green, 2);
                    
                    window.ShowImage(frame);
                    if (Cv2.WaitKey(1) >= 0) break;
                }
                
                // Cleanup implicit via using statement
                frame.Dispose();
                img.Dispose();
                window.Dispose();
            }
            capture.Dispose();
        }

        static void tryImageInference()
        {
            string img_path = "image.jpg";
            Mat image = Cv2.ImRead(img_path);
            if (image.Empty())
            {
                Console.WriteLine("Image not found: " + img_path);
                return;
            }

            using (var detector = new YoloDetector())
            {
                if (!detector.LoadModel(modelPath, deviceNum))
                {
                    Console.WriteLine("Model not loaded");
                    return;
                }
                Console.WriteLine("Model Loaded.");
                detector.SetThreshold(0.3f, 0.3f, 0.3f);

                int numObjects = detector.Detect(img_path, net_height, net_width);
                Console.WriteLine($"Objects detected: {numObjects}");

                if (numObjects > 0)
                {
                    var objects = detector.GetDetectedObjects(numObjects, image.Height, image.Width);
                    DrawResults(image, objects);
                }

                // Show result
                var displayImg = image.Clone();
                Cv2.Resize(displayImg, displayImg, new Size(image.Width / 4, image.Height / 4));
                Cv2.ImShow("image", displayImg);
                Cv2.WaitKey(0);
                Cv2.DestroyAllWindows();
                displayImg.Dispose();
            }
            image.Dispose();
        }


        static void DrawResults(Mat image, YoloObject[] objects)
        {
            if (objects == null || objects.Length == 0) return;

            // Use the segmentation map from the first object (shared)
            // Note: In the new C++ design, objects share the same internal seg_map pointer?
            // Yes, PopulateObjects uses 'total_seg_map.data' for all objects.
            
            IntPtr segMapPtr = objects[0].seg_map;
            if (segMapPtr == IntPtr.Zero) return;

            using (var segRegion = new Mat(image.Rows, image.Cols, MatType.CV_8UC3, segMapPtr))
            {
                double alpha = 0.8;
                double beta = 0.2;

                foreach (var obj in objects)
                {
                    Cv2.Rectangle(image, new Point((int)obj.left, (int)obj.top), 
                                  new Point((int)obj.right, (int)obj.bottom), Scalar.Red, 2);

                    // Blend segmentation
                    // We need to clip boxRect to image bounds
                    int x = Math.Max(0, (int)obj.left);
                    int y = Math.Max(0, (int)obj.top);
                    int w = Math.Min(image.Width - x, (int)(obj.right - obj.left));
                    int h = Math.Min(image.Height - y, (int)(obj.bottom - obj.top));

                    if (w > 0 && h > 0)
                    {
                        Rect boxRect = new Rect(x, y, w, h);
                        using (var roiImg = image[boxRect])
                        using (var roiSeg = segRegion[boxRect])
                        {
                            Cv2.AddWeighted(roiImg, alpha, roiSeg, beta, 0.0, roiImg);
                        }
                    }
                }
            }
        }

        static void Main(string[] args)
        {
            // Create a dummy file if not exists for testing logic? 
            // The user environment implies these files exist or are provided.
            
            //Console.WriteLine("Single Image Inference");
            //tryImageInference();

            //Console.WriteLine("Video Frame Inference");
            //tryFrameInference("video.mp4");

            Console.WriteLine("Webcam Inference");
            tryFrameInference("");
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.Internal;
using static System.Net.Mime.MediaTypeNames;

namespace YoloCShaprInference
{

    internal class Program
    {
        // Path to the model file.
        public static string modelPath = "yolov8s-seg.torchscript";

        // Dimensions of the input expected by the network.
        public static int net_height = 640;
        public static int net_width = 640;

        // Dimensions of the original image.
        private static int orgHeight;
        private static int orgWidth;

        // Enum to represent the computation device type.
        enum DeviceType
        {
            CPU = 0,
            CUDA = 1
        }

        // Device number to be used for inference.
        private static int deviceNum = (int)DeviceType.CUDA;

        // Thresholds for object detection and segmentation.
        private static float score_thresh = 0.3f;
        private static float iou_thresh = 0.3f;
        private static float seg_thresh = 0.3f;

        // Number of objects detected.
        private static int numObjects;

        // Array to hold detected objects.
        private static YoloObject[] YoloObjectArray;

        // Name of the DLL containing the inference code.
        private const string dll = "YoloV8DLLProject";

        // Reasons of using [StructLayout(LayoutKind.Sequential)]
        // 1. Defines Explicit Memory Layout
        // 2. Interop with Unmanaged Code
        // 3. Binary Compatibility
        // 4. Avoid Memory Corruption
        [StructLayout(LayoutKind.Sequential)]
        public struct YoloObject
        {
            // Bounding box coordinates, score, class ID, and pointer to the segmentation map.
            public float left;
            public float top;
            public float right;
            public float bottom;
            public float score;
            public float classID;
            public IntPtr seg_map;

            // Constructor for the YoloObject struct.
            public YoloObject(float left, float top, float right, float bottom, float score, float classID, IntPtr seg_map)
            {
                this.left = left;
                this.top = top;
                this.right = right;
                this.bottom = bottom;
                this.score = score;
                this.classID = classID;
                this.seg_map = seg_map;
            }
        };

        // Import SetDevice function from the DLL to set the computation device (CPU or GPU) for inference.
        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void SetDevice(int deviceNum);

        // Import SetThreshold to adjust detection and segmentation sensitivity.
        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void SetThreshold(float score_tresh, float iou_thresh, float seg_thresh);

        // Use LoadModel to load the model for object detection.
        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int LoadModel(string modelPath, int deviceNum);

        // PerformInference runs detection on a single image.
        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int PerformImagePathInference(string inputData, int net_height, int net_width);

        // PerformFrameInference handles detection on video or webcam streams.
        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int PerformFrameInference(IntPtr inputData, int net_height, int net_width);

        // PopulateObjectsArray formats detection results for C# use.
        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void PopulateYoloObjectsArray(IntPtr objects, int org_height, int org_width);

        // FreeResources clears memory used during detection to optimize performance.
        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void FreeAllocatedMemory();

        // AllocConsole opens a new console window for debugging output.
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AllocConsole();


        // Calls the PerformFrameInference function 
        // passing the current frame's data pointer and 
        // dimensions, and returns the number of detected objects.
        static unsafe int ReturnFramePerformance(Mat image, int net_height, int net_width)
        {
            numObjects = PerformFrameInference((IntPtr)image.DataPointer, net_height, net_width);
            return numObjects;
        }

        // Allocates an array for detected YOLO objects and populates it
        // by calling PopulateObjectsArray from the DLL.
        // The fixed keyword is used to pin the YoloObjectArray in memory,
        // providing an unchanging pointer to PopulateObjectsArray.
        static unsafe void RunPopulateYoloObjectsArray(int numProPosals, int org_height, int org_width)
        {
            // Initialize the array with the number of detections.
            YoloObjectArray = new YoloObject[numProPosals];

            // Pin the YoloObjectArray in memory.
            fixed (YoloObject* o = YoloObjectArray)
            {
                // Populate the array with detection data.
                PopulateYoloObjectsArray((IntPtr)o, org_height, org_width);
            }
        }

        // A function to perform frame-by-frame inference on video data, 
        // showing the results in a window.
        static void tryFrameInference(string videoPath)
        {
            VideoCapture capture; // OpenCV video capture object.

            // Initialize the VideoCapture object with a file path or a webcam index.
            if (!string.IsNullOrEmpty(videoPath) && System.IO.File.Exists(videoPath))
            {
                capture = new VideoCapture(videoPath); // Load video from path.
            }
            else
            {
                capture = new VideoCapture(0); // Default to the first webcam.
            }
            if (!capture.IsOpened()) // Check if the video source was successfully opened.
            {
                Console.WriteLine("Failed to open video source.");
                return;
            }

            // Initial setup for performing inference, including setting the inference device,
            // loading the model, and configuring detection thresholds.
            SetDevice(deviceNum);
            Console.WriteLine("Device Set _ " + " Device info : " + deviceNum);

            // Attempt to load the model for inference. If unsuccessful, exit the function.
            int loadModelVal = LoadModel(modelPath, deviceNum);
            if (loadModelVal == -1)
            {
                Console.WriteLine("Model not loaded");
                return;
            }
            Console.WriteLine("Model Loaded ?  : " + loadModelVal);
            // Set the thresholds for detection confidence, 
            // IoU (Intersection over Union), and segmentation mask confidence.
            SetThreshold(0.3f, 0.3f, 0.3f);
            Console.WriteLine("Threshold setting fishined .. ");

            {
                if (!capture.IsOpened())
                {
                    Console.WriteLine("Camera not found");
                    return;
                }
                // Begin capturing and processing video frames.
                // Create a window for displaying the video frames.
                var window = new Window("capture");
                // Initialize a Mat object to hold individual frames from the video.
                var frame = new Mat();
                // Initialize a Mat object for the processed frame.
                var img = new Mat();
                // Initialize a Mat object for holding segmentation maps. 
                var segRegion = new Mat();
                // Stopwatch for measuring frame processing time.
                Stopwatch stopwatch = new Stopwatch();
                while (true)
                {
                    // Read the next frame from the video capture device.
                    capture.Read(frame);
                    // Restart the stopwatch
                    stopwatch.Restart();
                    // Check if the captured frame is empty (end of video or error).
                    if (frame.Empty())
                    {
                        Console.WriteLine("Blank frame grabbed");
                        break; // Exit the loop if a blank frame is encountered.
                    }
                    // Resize the captured frame to match the input size expected by the network model.
                    Cv2.Resize(frame, img, new Size(net_width, net_height));
                    // Store the original dimensions of the frame for later use.
                    orgWidth = frame.Width;
                    orgHeight = frame.Height;
                    // Perform inference on the resized frame and obtain the number of detected objects.
                    numObjects = ReturnFramePerformance(img, net_height, net_width);
                    // Check the inference result and skip processing if no objects were detected or if an error occurred.
                    if (numObjects == 0)
                    {
                        // Skip the rest of the loop iteration and process the next frame.
                        continue;
                    }
                    else if (numObjects == -1)
                    {
                        Console.WriteLine("Error in inference");
                        continue;
                    }

                    // Populate the YoloObjectArray with detection data for further processing.
                    RunPopulateYoloObjectsArray(numObjects, orgHeight, orgWidth);

                    // Blend detected objects' segmentation maps onto the original frame for visualization.
                    double alpha = 0.8;
                    double beta = 0.2;
                    double gamma = 0.0;
                    Rect boxRect;
                    var obj2 = YoloObjectArray[0];
                    segRegion = new Mat(frame.Rows, frame.Cols, MatType.CV_8UC3, (IntPtr)obj2.seg_map);

                    // Iterate over each detected object to draw bounding boxes and blend segmentation maps.
                    for (int i = 0; i < YoloObjectArray.Length; i++)
                    {
                        var obj = YoloObjectArray[i];
                        // Draw a rectangle around the detected object.
                        Cv2.Rectangle(frame, new Point((int)(obj.left), (int)(obj.top)), new Point((int)(obj.right), (int)(obj.bottom)), Scalar.Red, 2);
                        // Define the region of the frame corresponding to the current object's bounding box.
                        boxRect = new Rect((int)(obj.left), (int)(obj.top), (int)(obj.right - obj.left), (int)(obj.bottom - obj.top));
                        //if (boxRect.Left < 0 || boxRect.Top < 0 || boxRect.Right > frame.Width || boxRect.Bottom > frame.Height)
                        //{
                        //    Console.WriteLine("BoxRect is out of image bounds");
                        //    return;
                        //}
                        //if (boxRect.Width <= 0 || boxRect.Height <= 0)
                        //{
                        //    Console.WriteLine("BoxRect has invalid dimensions");
                        //    return;
                        //}
                        // Blend the segmentation map for the detected object with the corresponding region of the frame.
                        Cv2.AddWeighted(frame[boxRect], alpha, segRegion[boxRect], beta, gamma, frame[boxRect]);

                    }
                    // Measure and display the frame processing time (FPS).
                    stopwatch.Stop(); // Stop the stopwatch.
                                      // Calculate frames per second (FPS).
                    double fps = 1000.0 / stopwatch.ElapsedMilliseconds;
                    // Display the FPS on the frame.
                    Cv2.PutText(frame, $"FPS: {fps:0.0}", new Point(10, 30), HersheyFonts.HersheySimplex, 1, Scalar.Green, 2);
                    window.ShowImage(frame); // Show the processed frame in the window.
                                             // Exit the loop if a key is pressed.
                    int key = Cv2.WaitKey(1);
                    if (key >= 0)
                    {

                        // Clean up and release resources after exiting the loop.
                        frame.Dispose();
                        segRegion.Dispose();
                        img.Dispose();
                        window.Dispose();
                        window.Close();
                        break;
                    }
                }
            }
            capture.Dispose();
            FreeAllocatedMemory(); // Free resources allocated by the DLL.
            Console.WriteLine("Resources Freed");
        }

        static void tryImageInference()
        {
            // Initial setup for performing inference, 
            // including setting the inference device, 
            // loading the model, and configuring detection thresholds.
            SetDevice(deviceNum);
            Console.WriteLine("Device Set _ " + " Device info : " + deviceNum);

            // Attempt to load the model for inference. If unsuccessful, exit the function.
            int loadModelVal = LoadModel(modelPath, deviceNum);
            if (loadModelVal == -1)
            {
                Console.WriteLine("Model not loaded");
                return;
            }
            Console.WriteLine("Model Loaded ?  : " + loadModelVal);
            string img_path = "image.jpg";
            // Read an Image
            Mat image = Cv2.ImRead(img_path);
            if (image.Empty())
            {
                Console.WriteLine("Image not found"); // Image reading occurs an error, then Exit.
                return;
            }
            // Store the original dimensions of the frame for later use.
            orgWidth = image.Width;
            orgHeight = image.Height;

            // Set the thresholds for detection confidence, 
            // IoU (Intersection over Union), and segmentation mask confidence.
            SetThreshold(0.3f, 0.3f, 0.3f);
            Console.WriteLine("Threshold setting fishined .. ");
            // Perform inference on the image
            numObjects = PerformImagePathInference(img_path, net_height, net_width);

            // Check the inference result and skip processing if no objects were detected or if an error occurred.
            if (numObjects == 0)
            {
                // Skip the rest of the loop iteration
                Console.WriteLine("No objects detected");
                return;
            }
            else if (numObjects == -1)
            {
                Console.WriteLine("Error in inference");
                return;
            }
            Console.WriteLine("PerformInference Implemented..");
            Console.WriteLine("numObjects : " + numObjects);

            // Populate the YoloObjectArray with detection data for further processing. 
            RunPopulateYoloObjectsArray(numObjects, orgHeight, orgWidth);
            Console.WriteLine("RunPopulateObjectsArray Implemented..");

            Console.WriteLine($"Num Objects : {numObjects}");

            // Blend detected objects' segmentation maps onto the original frame for visualization.
            double alpha = 0.8;
            double beta = 0.2;
            double gamma = 0.0;
            Rect boxRect;
            var obj2 = YoloObjectArray[0];
            // get seg_map's memory
            IntPtr ptr = (IntPtr)obj2.seg_map;
            int rows = image.Rows;
            int cols = image.Cols;
            int type = MatType.CV_8UC3;
            var segRegion = new Mat(rows, cols, type, ptr);

            // Iterate over each detected object to draw bounding boxes and blend segmentation maps.
            for (int i = 0; i < YoloObjectArray.Length; i++)
            {
                var obj = YoloObjectArray[i];
                // Draw a rectangle around the detected object.
                Cv2.Rectangle(image, new Point((int)(obj.left), (int)(obj.top)), new Point((int)(obj.right), (int)(obj.bottom)), Scalar.Red, 2);
                // Define the region of the frame corresponding to the current object's bounding box.
                boxRect = new Rect((int)(obj.left), (int)(obj.top), (int)(obj.right - obj.left), (int)(obj.bottom - obj.top));

                // Check if the bounding box is within the bounds of the original image.
                if (boxRect.Left < 0 || boxRect.Top < 0 || boxRect.Right > image.Width || boxRect.Bottom > image.Height)
                {
                    // Exit the function if the bounding box is out of bounds, preventing errors.
                    Console.WriteLine("BoxRect is out of image bounds");
                    return;
                }
                // Validate the dimensions of the bounding box to ensure they are positive.
                if (boxRect.Width <= 0 || boxRect.Height <= 0)
                {
                    // Exit the function if the bounding box has invalid dimensions.
                    Console.WriteLine("BoxRect has invalid dimensions");
                    return;
                }
                // Print details of the bounding box and the image for debugging or informational purposes.
                Console.WriteLine(boxRect.Left + " " + boxRect.Right + " " + boxRect.TopLeft + " " + " BoxRect : " + boxRect.Size.Height + " " + boxRect.Size.Width + " Image " + image.Height + " " + image.Width + " segRegion :  " + segRegion.Height + " " + image.Width);

                // Blend the segmentation map for the detected object with the corresponding region of the frame.
                Cv2.AddWeighted(image[boxRect], alpha, segRegion[boxRect], beta, gamma, image[boxRect]);

            }
            // Resize the processed image to a quarter of its original size for display.
            // This is hard-coded and may be adjusted depending on display or performance requirements.
            var imsize = new Size(image.Width / 4, image.Height / 4); // hard coding
            Cv2.Resize(image, image, imsize);

            // Display the processed image in a window named "image".
            Cv2.ImShow("image", image);

            // Wait indefinitely for a key press before proceeding. 
            // This allows users to view the processed image.
            int keyPressed = Cv2.WaitKey(0); // This will wait indefinitely for a key press

            // Destroy all OpenCV windows created during the execution to free resources.
            Cv2.DestroyAllWindows();

            // Dispose of the Mat objects holding the original and segmentation images to free memory.
            image.Dispose();
            segRegion.Dispose();

            // Call a function to free any additional resources used during processing,
            // such as loaded models or temporary data.
            FreeAllocatedMemory();
            Console.WriteLine("Resources Freed");
        }



        static void Main(string[] args)
        {
            // Begin inference on a single image.
            //Console.WriteLine("Single Image Inference");
            //tryImageInference(); // Processes a single image through the model.

            // Start processing a video file for inference frame by frame.
            //Console.WriteLine("Video Frame Inference");
            //tryFrameInference("video.mp4"); // Applies model to each frame of "video.mp4".

            // Switch to real-time inference using webcam footage.
            // A Webcam should be prepared beforehand.
            Console.WriteLine("Webcam Inference");
            // Uses webcam stream for model inference, interpreting "" to select default camera.
            tryFrameInference("");

        }
    }
}

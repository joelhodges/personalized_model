using System.IO;
using System.Runtime.InteropServices;
using OpenCvSharp;

class VideoProcessor
{
    const int NUM_SAMPLES = 20; // Number of images to use to train the model
    const int SIZE = 128; // Image size, in pixels
    
    // Allowed extensions for videos and images
    string[] VIDEO_EXTENSIONS = new string[2]{".mov",".mp4"};
    string[] IMAGE_EXTENSIONS = new string[3]{".jpg",".jpeg",".png"};

    public VideoProcessor(string rootFolder, string modelFile)
    {
        string[] folders = Directory.GetDirectories(rootFolder); // Stores all folder names within the database

        System.DateTime modelTime = GetUpdateTime(modelFile); // The last update time of the model
        List<string>need_proc = new List<string>(); // Stores all folders that need to be reprocessed

        // Look through each folder in the database
        foreach (string folder in folders)
        {
            // Gets the last time the Gallery was modified for the current object
            System.DateTime galleryModTime = GetUpdateTime(folder + "\\Gallery");
            
            // Gallery only needs reprocessing if it has been updated since the last time the model was trained
            if (modelTime < galleryModTime)
            {
                need_proc.Add(folder);
            }
        }
        
        // Run downsizing for each folder that needs processing
        foreach (string folder in need_proc){
            Downsize(folder);
        }  
    }

    public System.DateTime GetUpdateTime(string folder)
    {
        System.DateTime max_mod_time = Directory.GetLastWriteTime(folder);
        string[] files = Directory.GetFiles(folder, "*ProfileHandler.cs", SearchOption.AllDirectories);

        foreach (string file in files)
        {
            if(File.GetLastWriteTime(file) > max_mod_time){
                max_mod_time = File.GetLastWriteTime(file);
            }    
        }
        return max_mod_time;
    }

    public void Downsize(string folder)
    {
        string gallery = folder + "\\Gallery"; // The path to the Gallery folder for the current object
        string downsized = folder + "\\Model"; // The path to the Model folder for the current object
        int step = GetStepSize(gallery);
        int count = 0;
        
        // Remove all images from the current Model folder
        foreach (string file in Directory.GetFiles(downsized)){
            File.Delete(file);
        }
        
        // Add new images to the Model folder
        foreach (string file in Directory.GetFiles(gallery)){
            int fileType = 0; //0 for nothing, 1 for image, 2 for video

            foreach (string ext in VIDEO_EXTENSIONS){
                if (file.ToLower().Contains(ext)){
                    fileType = 2;
                }
            }

            if(fileType == 0){
                foreach (string ext in IMAGE_EXTENSIONS){
                    if (file.ToLower().Contains(ext)){
                        fileType = 1;
                    }
                }
            }
            
            // Process videos
            if (fileType == 2){
                VideoCapture vidcap = new VideoCapture(file);
                Mat currentFrame = new Mat();
                bool success = vidcap.Read(currentFrame);
                while (success){
                    
                    // Resize current frame and write to the Model folder
                    Mat resized = new Mat();
                    Cv2.Resize(currentFrame, resized, new Size(SIZE, SIZE), interpolation: InterpolationFlags.Area);
                    Cv2.ImWrite(downsized + "\\image_" + count.ToString() + ".jpg", resized);
                    count++;
                
                    // Take a certain number of steps to get to approximately NUM_SAMPLES images
                    for (int i = 0; i < step; i++){
                        success = vidcap.Read(currentFrame);
                    }
                }
            }
                    
            // Process images
            else if (fileType==1){
                
                // Resize image and write to the Model folder
                Mat image = Cv2.ImRead(gallery + '/' + file);
                Mat resized = new Mat();
                Cv2.Resize(image, resized, new Size(SIZE, SIZE), interpolation: InterpolationFlags.Area);
                Cv2.ImWrite(downsized + "\\image_" + count.ToString() + ".jpg", resized);
                count++;
            }
        }
    }

    public int GetStepSize(string folder)
    {
        int frameCount = 0;
        int fileType; //0 for nothing, 1 for image, 2 for video
        int neededSamples = NUM_SAMPLES;
        
        // Look through each file in the folder (all should be image or video format)
        foreach (string file in Directory.GetFiles(folder)){

            Console.WriteLine(file);

            fileType = 0;
            foreach (string ext in VIDEO_EXTENSIONS){
                if (file.ToLower().Contains(ext)){
                    fileType = 2;
                }
            }

            if(fileType == 0){
                foreach (string ext in IMAGE_EXTENSIONS){
                    if (file.ToLower().Contains(ext)){
                        fileType = 1;
                    }
                }
            }
            
            // Calculate number of frames in each video and add to the total frame count
            if (fileType == 2){
                VideoCapture vidcap = new VideoCapture(file);
                frameCount = frameCount + vidcap.FrameCount;
            }
            // All images are included in the model, so less samples are needed from the videos for each image found
            else if(fileType == 1){
                neededSamples--;
            }
        }
        return frameCount/neededSamples;
    }
}

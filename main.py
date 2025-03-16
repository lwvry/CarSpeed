import argparse
from VideoProcessor import VideoProcessor
import cv2

def main():
    parser = argparse.ArgumentParser(description="Process video to analyze car speeds.")
    parser.add_argument("--video_path", type=str, required=False, help="Path to the input video file.")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the processed video.")
    parser.add_argument("--drone_model", type=str, choices=["DJI mini 4 pro", "DJI air 2s"], required=False, help="Drone model used for video recording.")
    parser.add_argument("--start_altitude", type=float, required=False, help="Starting altitude of the drone (in meters).")
    
    args = parser.parse_args()
    video_path = "/Users/maciejlower/Downloads/OneDrive_3_7/DJI_20240709125210_0005_D.MP4"
    output_path = "/Users/maciejlower/Downloads/OneDrive_3_7/MainOut10-1-poprawione.MP4"
    drone_model = "DJI mini 4 pro"
    start_altitude = None
    try:
        video_processor = VideoProcessor(
            video_path, 
            drone_model, 
            start_altitude, 
            model_path="models/drone7liten-obb-dota_and_data22.pt"
        )
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                video_processor.fps,
                (video_processor.frame_width, video_processor.frame_height)
            )
        
        print("Starting video processing...")
        total_frames = video_processor.get_total_frame_count()
        for frame_count in range(total_frames):
            frame, is_frame_available = video_processor.process_frame()
            if not is_frame_available:
                break
            
            if output_writer:
                output_writer.write(frame)

            print(f"Processed frame {frame_count + 1}/{total_frames}")
        
        if output_writer:
            output_writer.release()

        print("Video processing completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

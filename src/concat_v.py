from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def concat_videos_and_delete(folders = ["Greedy/", "Softmax/", "DQN/"],
                             name_video = "combined_video.mp4"):



    # Combine all the videos
    for video_folder in folders:
        video_folder = "video_" + video_folder


        # Get all video files from the folder
        video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

        # Load the video clips
        video_clips = [VideoFileClip(f) for f in video_files]

        # Concatenate all video clips into one
        final_clip = concatenate_videoclips(video_clips)

        # Write the final video to a file
        final_clip.write_videofile(video_folder+name_video, codec="libx264")

    # delete all the others
    for video_folder in folders:
        video_folder = "video_" + video_folder
        # Delete all files except for the concatenated one
        for video_file in os.listdir(video_folder):
            if not( name_video in video_file):
                file_path = os.path.join(video_folder, video_file)
                os.remove(file_path)
                print(f'Deleted: {file_path}')
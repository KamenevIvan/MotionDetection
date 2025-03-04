import cv2
import os
import argparse


SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]


def create_video(image_folder, output_video, fps=25):
    images = [
        img
        for img in os.listdir(image_folder)
        if os.path.splitext(img)[1].lower() in SUPPORTED_IMAGE_FORMATS
    ]
    images.sort()

    if not images:
        print(f"Folder {image_folder} doesn't have images")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (width, height))
        if frame is not None:
            video.write(frame)
        else:
            print(f"Error reading image: {img_path}")

    video.release()
    print(f"Video save in {output_video}")


def main(folder_path, fps):
    create_video(folder_path, os.path.join(folder_path, "output_video.mp4"), fps)

    maps_folder = os.path.join(folder_path, "maps")
    if os.path.exists(maps_folder):
        create_video(maps_folder, os.path.join(folder_path, "maps_video.mp4"), fps)
    else:
        print(f"Folder {maps_folder} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="VideoBuilder", description="Use: python script.py <FPS> <Path_to_folder>"
    )

    parser.add_argument("fps", type=int)
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    main(args.path, args.fps)

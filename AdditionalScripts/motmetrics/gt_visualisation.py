import argparse
import csv
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnnotationFormatError(Exception):
    """Custom exception for unsupported annotation formats."""
    pass


class VideoProcessor:
    """Handles video processing and annotation visualization."""

    def __init__(self):
        self.cap = None
        self.writer = None
        self.video_info = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_resources()

    def release_resources(self):
        """Release all video resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.writer and self.writer.isOpened():
            self.writer.release()
        cv2.destroyAllWindows()

    def open_video(self, video_path: str) -> Dict[str, Any]:
        """Open video file and return its properties."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        self.video_info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC))
        }
        return self.video_info

    def create_output_writer(self, output_path: str) -> None:
        """Create video writer for output."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.video_info['fps'],
            (self.video_info['width'], self.video_info['height'])
        )

    def draw_bbox(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        track_id: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_id: bool = True
    ) -> np.ndarray:
        """Draw bounding box on frame with optional ID label."""
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        if show_id and track_id is not None:
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness
            )
        return frame


class AnnotationParser:
    """Parse different annotation file formats."""

    @staticmethod
    def detect_format(file_path: str) -> str:
        """Detect annotation file format."""
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        parts = first_line.split(',')
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            return "format1"  # frame_id,track_id,x,y,w,h,...
        elif len(parts) >= 4 and all(p.lstrip('-').isdigit() for p in parts[:4]):
            return "format2"  # x,y,w,h,score (sequential frames)
        else:
            raise AnnotationFormatError("Unsupported annotation format")

    @staticmethod
    def parse_file(file_path: str) -> List[Tuple[int, Optional[int], int, int, int, int]]:
        """Parse annotation file based on detected format."""
        format_type = AnnotationParser.detect_format(file_path)
        annotations = []

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row:
                    continue

                try:
                    if format_type == "format1":
                        frame_id = int(row[0])
                        track_id = int(row[1])
                        x, y, w, h = map(int, row[2:6])
                    else:  # format2
                        frame_id = i + 1  # 1-based frame numbering
                        track_id = None
                        x, y, w, h = map(int, row[:4])
                    
                    annotations.append((frame_id, track_id, x, y, w, h))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping malformed line {i+1}: {row}. Error: {e}")
                    continue

        return annotations


def process_video(
    video_path: str,
    annotation_path: str,
    output_path: str,
    show_ids: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """Main processing function to draw annotations on video."""
    try:
        # Validate paths
        video_path = Path(video_path)
        annotation_path = Path(annotation_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        # Parse annotations
        logger.info("Parsing annotation file...")
        annotations = AnnotationParser.parse_file(str(annotation_path))
        if not annotations:
            raise ValueError("No valid annotations found in the file")

        # Process video
        with VideoProcessor() as processor:
            # Setup video processing
            video_info = processor.open_video(str(video_path))
            processor.create_output_writer(str(output_path))

            logger.info(f"Processing video: {video_path.name}")
            logger.info(f"Resolution: {video_info['width']}x{video_info['height']}")
            logger.info(f"FPS: {video_info['fps']:.2f}, Frames: {video_info['total_frames']}")

            current_frame = 1
            annotation_index = 0
            total_annotations = len(annotations)

            while True:
                ret, frame = processor.cap.read()
                if not ret:
                    break

                # Draw all annotations for current frame
                while (annotation_index < total_annotations and 
                       annotations[annotation_index][0] == current_frame):
                    frame_id, track_id, x, y, w, h = annotations[annotation_index]
                    
                    # Skip invalid boxes
                    if w > 0 and h > 0:
                        processor.draw_bbox(
                            frame, x, y, w, h,
                            track_id=track_id if show_ids else None,
                            color=color,
                            thickness=thickness
                        )
                    annotation_index += 1

                # Write processed frame
                processor.writer.write(frame)

                # Log progress every 5% or 100 frames
                if current_frame % max(100, video_info['total_frames'] // 20) == 0:
                    progress = current_frame / video_info['total_frames'] * 100
                    logger.info(f"Progress: {progress:.1f}% ({current_frame}/{video_info['total_frames']})")

                current_frame += 1

        logger.info(f"Successfully processed {current_frame-1} frames")
        logger.info(f"Output saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise


def main():
    """Command line interface for the script."""
    parser = argparse.ArgumentParser(
        description='Visualize ground truth bounding boxes on videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input', 
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '-a', '--annotations',
        required=True,
        help='Path to annotation file'
    )
    parser.add_argument(
        '-o', '--output',
        default='output.mp4',
        help='Path to output video file'
    )
    parser.add_argument(
        '--no-ids',
        action='store_false',
        dest='show_ids',
        help='Disable showing track IDs on boxes'
    )
    parser.add_argument(
        '--color',
        nargs=3,
        type=int,
        default=[0, 255, 0],
        help='Bounding box color in BGR format (e.g., 0 255 0 for green)'
    )
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Bounding box line thickness'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        process_video(
            video_path=args.input,
            annotation_path=args.annotations,
            output_path=args.output,
            show_ids=args.show_ids,
            color=tuple(args.color),
            thickness=args.thickness
        )
    except Exception as e:
        logger.critical(f"Script failed: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
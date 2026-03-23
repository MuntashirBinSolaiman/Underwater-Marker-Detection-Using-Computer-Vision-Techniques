import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.sam2_video_predictor import SAM2VideoPredictor
from visualizer import Visualizer
from frame_loader import FrameLoader


class SAM2Processor:
    def __init__(self, device, video_directory, visualizer=None, frame_loader=None):
        self.device = device
        self.video_directory = video_directory
        self.visualizer = visualizer if visualizer else Visualizer()
        self.frame_loader = frame_loader if frame_loader else FrameLoader(video_directory)

        self.predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large", device=device)
        self.image_files = self.frame_loader.get_image_files()
        self.inference_state = self.predictor.init_state(video_path=video_directory)

    def apply_clicks_to_predictor(self, frame_index, object_id, click_points, click_labels):
        return self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_index,
            obj_id=object_id,
            points=click_points,
            labels=click_labels,
        )

    def display_mask_on_frame(self, frame_index, points, labels, mask, object_id, show=True):
        image_path = os.path.join(self.video_directory, self.image_files[frame_index])
        image = Image.open(image_path)

        if show:
            plt.figure(figsize=(9, 6))
            plt.title(f"Frame {frame_index}")
            plt.imshow(image)
            self.visualizer.show_points(points, labels, plt.gca())
            self.visualizer.show_mask(mask, plt.gca(), obj_id=object_id)
            plt.show()

    @staticmethod
    def convert_mask_logits_to_numpy(mask_logits_tensor):
        return (mask_logits_tensor > 0.0).cpu().numpy()

    def propagate_and_segment_video(self):
        segments = {}
        for frame_index, object_ids, mask_logits in self.predictor.propagate_in_video(self.inference_state):
            segments[frame_index] = {}
            for i in range(len(object_ids)):
                mask = self.convert_mask_logits_to_numpy(mask_logits[i])
                segments[frame_index][object_ids[i]] = mask
        return segments

    def visualize_segments(self, segments, stride=1, show=True):
        if show:
            plt.close("all")
        for index in range(0, len(self.image_files), stride):
            image_path = os.path.join(self.video_directory, self.image_files[index])
            image = Image.open(image_path)

            if show:
                plt.figure(figsize=(6, 4))
                plt.title(f"Frame {index}")
                plt.imshow(image)

            if index in segments:
                for object_id, mask in segments[index].items():
                    self.visualizer.show_mask(mask, plt.gca(), obj_id=object_id)

            if show:
                plt.show()

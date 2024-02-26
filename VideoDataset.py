from torch.utils.data import Dataset, DataLoader
import cv2
import os
import re
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, directory, transform=None, target_frames=16,model_type='3DCNN'):
        self.directory = directory
        self.transform = transform
        self.target_frames = target_frames
        self.model_type = model_type
        self.videos = [os.path.join(directory, x) for x in os.listdir(directory) if x.endswith('.mp4')]
        self.labels = []

        # Modified regex pattern to match additional characters
        pattern = r"([a-zA-Z_]+)(?: \(.*\))?_\d+\.mp4"
        # Mapping from activity names to integers
        activity_to_idx = {
            'sitting_still': 0, 'eating': 1, 'fetching_an_object': 2, 'placing_an_object': 3,
            'reading_magazine':4, 'using_multimedia_display':5, 'talking_on_phone' : 6, 'writing': 7,
            'pressing_automation_button': 8, 'putting_on_jacket': 9, 'drinking': 10, 'fastening_seat_belt': 11,
            'taking_off_jacket': 12, 'looking_or_moving_around': 13, 'opening_bottle': 14, 'interacting_with_phone': 15,
            'working_on_laptop': 16,'reading_newspaper' : 17,
             'closing_bottle': 18, 'opening_laptop': 19
        }

        # Use regex to extract the activity name and convert it to an integer index
        for x in self.videos:
            match = re.match(pattern, os.path.basename(x))
            if match is not None:
                activity_name = match.group(1)
                # Append the integer index that corresponds to the activity name
                self.labels.append(activity_to_idx[activity_name])
            else:
                # Handle unexpected filename formats if necessary
                raise ValueError(f"Filename {x} does not match expected pattern.")
        self.label_to_index = {label: index for index, label in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.model_type == '2DCNN':
          video_path = self.videos[idx]
          frames = self.load_video(video_path, self.target_frames)

          # Select a frame, e.g., the first one
          frame = frames[0]

          # Convert frame to PIL Image and apply transformation
          if self.transform:
              frame = self.transform(Image.fromarray(frame))
          else:
              frame = transforms.ToTensor()(Image.fromarray(frame))
          pass

        elif self.model_type == '3DCNN':
            video_path = self.videos[idx]
            frames = self.load_video(video_path, self.target_frames)

            if self.transform:
                frames = [self.transform(frame) for frame in frames]

            frames_tensor = torch.stack(frames)  # Should be [depth, height, width]
            ##frames_tensor = frames_tensor.unsqueeze(1)  # Add channel dim: [depth, channels, height, width]
            pass
        label_index = self.label_to_index[self.labels[idx]]
        return frames_tensor, label_index


    def load_video(self, video_path, target_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, frame_count - 1, target_frames, dtype=np.int64)
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = Image.fromarray(frame)
                frames.append(frame)
        cap.release()
        # Pad frames if necessary
        while len(frames) < target_frames:
            padding_frame = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
            frames.append(padding_frame)
        return frames

# Assuming that the transformations are correct and only applicable to single frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

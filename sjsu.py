
import io
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model
import random


class ObjectDetector:
    def __init__(self,model_name):
        """Initialize the object detector."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model and processor
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def florence2(self, task_prompt, image, text_input=None):
        """
        Calling the Microsoft Florence-2 model with GPU support.
        """
        # Use the device initialized in the class
        device = self.device
        print(f"Using device: {device}")

        # Move model to the correct device
        self.model.to(device)

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        # Move image input to the correct device
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU/CPU

        # Run the model on the correct device
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        return parsed_answer

    def plot_bbox(self, image, data):
        """Plot bounding boxes on the image."""
        fig, ax = plt.subplots()
        ax.imshow(image)
        for bbox, label in zip(data['bboxes'], data['labels']):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        ax.axis('off')
        plt.show()

    def detect_objects(self, processor, model, image):
      task_prompt = "<OD>"  # Object Detection task prompt
      return self.florence2(task_prompt, image)




    def add_noise(self, image, noise_type):
        mean = 5
        var = 40
        salt_vs_pepper = 0.4
        amount = 0.02

        if noise_type == "gaussian":
            row, col, ch = image.shape
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)

            # Ensure pixel values remain in valid range
            noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

        elif noise_type == "salt_pepper":
            noisy = image.copy()
            num_salt = int(amount * image.size * salt_vs_pepper)
            num_pepper = int(amount * image.size * (1.0 - salt_vs_pepper))

            # Add salt (white pixels)
            coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
            noisy[tuple(coords_salt)] = 255

            # Add pepper (black pixels)
            coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
            noisy[tuple(coords_pepper)] = 0

        elif noise_type == "poisson":
            vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
            noisy = np.random.poisson(image.astype(np.float32) * vals) / float(vals)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        else:
            return image  

        return noisy
    
    def process_detected_objects(self, detected_objects, raw_frame_filename, answer):
        """Process all detected objects and store data."""
        all_frame_data = []
        bboxes = detected_objects.get('<OD>', {}).get('bboxes', [])
        labels = detected_objects.get('<OD>', {}).get('labels', [])
        
        for bbox, label in zip(bboxes, labels):
            all_frame_data.append({
                'raw_frame_path': raw_frame_filename,
                'label': answer,   
                'bbox': bbox
            })
        
        return all_frame_data  # List of all detected objects with labels and bounding boxes

    def process_video(self, video_path, output_frame_dir, answer, skip_frames, rotate=None):
        cap = cv2.VideoCapture(video_path)
        frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video resolution: {frame_width}x{frame_height}")

        all_frame_data = []
        frame_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop when the video ends


            if rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Resize frame to 640x640 before inference
            frame_resized = cv2.resize(frame, (640, 640))

            if frame_counter % skip_frames == 0:
                frame_filename = os.path.join(output_frame_dir, f"{answer}_{frame_counter:04d}.jpg")
                cv2.imwrite(frame_filename, frame_resized)

                # Generate different noise variations
                noise_types = ["gaussian", "salt_pepper", "poisson"]
                noise_filenames = []
                noise_images = []

                for noise in noise_types:
                    noisy_image = self.add_noise(frame_resized, noise)
                    noisy_filename = os.path.join(output_frame_dir, f"{answer}_{noise}_{frame_counter:04d}.jpg")
                    cv2.imwrite(noisy_filename, noisy_image)
                    noise_filenames.append(noisy_filename)
                    noise_images.append(noisy_image)

                # Convert original image to PIL for inference
                pil_original = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

                # Run inference only on the original frame
                detected_objects = self.detect_objects(self.processor, self.model, pil_original)

                # Process detected objects and get bounding boxes
                frame_data = self.process_detected_objects(detected_objects, frame_filename, answer)
                all_frame_data.extend(frame_data)

                # Plot bounding boxes on the original image
                self.plot_bbox(pil_original, {
                    'bboxes': [d['bbox'] for d in frame_data],
                    'labels': [d['label'] for d in frame_data]
                })

                # Apply the same bounding boxes to noisy images
                for noisy_filename, noisy_image in zip(noise_filenames, noise_images):
                    pil_noisy = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))

                    # Store noisy frame data with original bounding boxes
                    noisy_frame_data = [
                        {**d, 'filename': noisy_filename} for d in frame_data  # Copy original data, update filename
                    ]
                    all_frame_data.extend(noisy_frame_data)

                    # Plot bounding boxes on noisy images
                    self.plot_bbox(pil_noisy, {
                        'bboxes': [d['bbox'] for d in frame_data],
                        'labels': [d['label'] for d in frame_data]
                    })

            frame_counter += 1  # Increment frame counter only once per loop

        cap.release()
        return all_frame_data


    def save_detection_data(self, all_frame_data, json_path, answer):
        filtered_frames_data = {}
        
        for frame_data in all_frame_data:
            frame_path, bbox = frame_data['raw_frame_path'], frame_data['bbox']
            if frame_path not in filtered_frames_data:
                filtered_frames_data[frame_path] = {'label': answer, 'bboxes': []}
            filtered_frames_data[frame_path]['bboxes'].append(bbox)


        with open(json_path, "w") as json_file:
            json.dump(filtered_frames_data, json_file, indent=4)
            print(f"Saved {len(filtered_frames_data)} frames to {json_path}")




            




            
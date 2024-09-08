# import os
# import cv2
# import albumentations as A
# from tqdm import tqdm

# def create_augmentations():
#     return A.Compose([
#         A.OneOf([
#             A.HorizontalFlip(p=1),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
#             A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=1),
#             A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
#             A.GaussianBlur(blur_limit=(3, 7), p=1),
#             A.ToGray(p=1),
#         ], p=1),
#     ])

# def apply_random_augmentation(image_path, save_path, transform):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image not found or cannot be opened: {image_path}")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     augmented = transform(image=image)['image']
#     augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
#     cv2.imwrite(save_path, augmented)

# def process_directory(input_dir, output_dir):
#     transform = create_augmentations()

#     # Create the output directory if it does not exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Walk through the input directory to process each file
#     for root, dirs, files in os.walk(input_dir):
#         for dir in dirs:
#             dir_path = os.path.join(root, dir)
#             out_dir_path = os.path.join(output_dir, os.path.relpath(dir_path, input_dir))
#             os.makedirs(out_dir_path, exist_ok=True)
            
#             # Process each file within the current directory
#             for file in os.listdir(dir_path):
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     input_path = os.path.join(dir_path, file)
#                     output_path = os.path.join(out_dir_path, file)

#                     # Apply a random augmentation
#                     try:
#                         apply_random_augmentation(input_path, output_path, transform)
#                     except Exception as e:
#                         print(f"Failed to process {input_path}: {str(e)}")

# # Specify your directory paths here
# input_directory = '/raid/home/adityalk/sid/tarun/clip/data/frames_each_audio'
# output_directory = '/raid/home/adityalk/sid/tarun/clip/data/frames_each_audio_aug'

# process_directory(input_directory, output_directory)



import os
import cv2
import albumentations as A
from tqdm import tqdm

def create_augmentations():
    return A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.ToGray(p=1),
        ], p=1),
    ])

def apply_random_augmentation(image_path, save_path, transform):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or cannot be opened: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    augmented = transform(image=image)['image']
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    cv2.imwrite(save_path, augmented)

def process_directory(input_dir, output_dir):
    transform = create_augmentations()

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate total number of images for tqdm progress bar
    total_images = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_images += 1

    # Reinitialize the loop to apply transformations with tqdm
    progress_bar = tqdm(total=total_images, desc="Processing Images")
    
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            out_dir_path = os.path.join(output_dir, os.path.relpath(dir_path, input_dir))
            os.makedirs(out_dir_path, exist_ok=True)
            
            # Process each file within the current directory
            for file in os.listdir(dir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_path = os.path.join(dir_path, file)
                    output_path = os.path.join(out_dir_path, file)

                    # Apply a random augmentation and update progress
                    try:
                        apply_random_augmentation(input_path, output_path, transform)
                        progress_bar.update(1)  # Update the progress bar per file processed
                    except Exception as e:
                        print(f"Failed to process {input_path}: {str(e)}")

    progress_bar.close()

# Specify your directory paths here
input_directory = '/raid/home/adityalk/sid/tarun/clip/data/frames_each_audio'
output_directory = '/raid/home/adityalk/sid/tarun/clip/data/frames_each_audio_aug'

process_directory(input_directory, output_directory)

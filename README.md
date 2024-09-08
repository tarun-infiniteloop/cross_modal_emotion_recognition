
# Cross-Modal Emotion Recognition: A Unified Framework for Video, Audio, and Text Integration

This project presents a novel approach to multimodal emotion recognition, integrating video, audio, and text data. Our method generates a unified vector space using **multitask learning**, **contrastive learning**, and **autoencoder techniques**. The system outperforms current state-of-the-art methods on the IEMOCAP benchmark dataset.

![Alt text](images/Fig1_4.png)

![Alt text](images/Fig2_2.png)

## Requirements

- Python 3.x
- Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```
  Requirements include:
  - `ffmpeg`
  - `transformers`
  - `torch`
  - `librosa`
  - `albumentations`
  - `sentence-transformers`
  - `tqdm`
  - `pyttsx3`
  - `Vosk`
  - `soundfile`
  - `Pillow`

Ensure that you also have `ffmpeg` installed for frame extraction:

```bash
ffmpeg-git-20240301-amd64-static/ffmpeg
```

## Setup Instructions

### 1. Install `ffmpeg`

Make sure to install `ffmpeg`, which is required for video frame extraction:

```bash
# Example for ffmpeg installation
ffmpeg-git-20240301-amd64-static/ffmpeg
```

### 2. Download the IEMOCAP Dataset

Manually download the IEMOCAP dataset from the official website. Once downloaded, place it in the appropriate directory inside this repository.

### 3. Preprocessing

Run `model_preprocessor_download.py` to download the CLIP model. This will download `openai/clip-vit-base-patch32` and create folders for both the model and the processor.

```bash
python model_preprocessor_download.py
```

### 4. Data Preparation

Navigate to the `data_prep` folder and run the following preprocessing scripts in sequence. These scripts generate the necessary embeddings and augmentations required for the model:

```bash
cd data_prep
```

1. **Text Augmentation**: 
   Translates sentences through multiple languages (English → French → German → English) using MarianMT models.

   ```bash
   python Text_translation.py
   ```

2. **Audio Augmentation**:
   Performs speech-to-text, translation, and text-to-speech conversion.

   ```bash
   python audioWAV_file_Augmentation.py
   ```

3. **MFCC Feature Extraction**:
   Extracts MFCC features from audio files in the IEMOCAP dataset.

   ```bash
   python audio_features_mfcc.py
   ```

4. **Frame Extraction**:
   Extracts video frames for each audio file based on timestamps.

   ```bash
   python data_prep_frame_each_audio_2.py
   ```

5. **Image Embeddings**:
   Extracts embeddings from video frames using the pre-trained CLIP model.

   ```bash
   python frames_emd_each_audio.py
   ```

6. **Text Embeddings**:
   Generates text embeddings using the `SentenceTransformer` model (all-mpnet-base-v2).

   ```bash
   python text_emb_generation.py
   ```

7. **Label Mapping**:
   Maps categorical labels in the CSV file (e.g., 'neu', 'ang', etc.) to numerical values.

   ```bash
   python csv_label_number_mapping.py
   ```

8. **Image Augmentation**:
   Augments the extracted frames using transformations like flips, contrast adjustments, and rotations.

   ```bash
   python image_augmentation.py
   ```

### 5. Model Training

Once all the preprocessing is done, run the `main.py` script to train the model using all the generated embeddings (video, text, audio):

```bash
python main.py
```

## Folder and File Structure

### Root Folder

```
Cross-Modal-Emotion-Recognition/
│
├── Reference_Audio_Files/           # Sample reference audio files (5 male, 5 female)
├── data_preprocess/                 # Contains preprocessing scripts for data
│   ├── Text_translation.py          # Performs text data augmentation via translations
│   ├── audioWAV_file_Augmentation.py# Speech-to-text, translation, and text-to-speech
│   ├── audio_features_mfcc.py       # Extracts MFCC features from audio files
│   ├── aug_audio_mfcc.py            # Processes augmented audio for MFCC extraction
│   ├── csv_label_number_mapping.py  # Maps text labels to numerical labels
│   ├── data_prep_frame_each_audio_2.py # Extracts frames for each audio
│   ├── frames_emd_each_audio.py     # Generates frame embeddings using CLIP
│   ├── image_augmentation.py        # Augments images/frames with transformations
│   ├── text_emb_generation.py       # Generates text embeddings for sentences
│   ├── text_emb_generation_testing_sbertnet.py # Demo of text to embedding generation
│
├── images/                          # Images used in the research paper
│   ├── Fig1_4.png                   # Example figure 1
│   ├── Fig2_2.png                   # Example figure 2
│
├── model_preprocessor_download.py    # Downloads CLIP model and processor
├── main.py                          # The main script to train the model
├── iemocap.csv                      # The IEMOCAP dataset CSV file
├── README.md                        # Readme file with instructions
├── data/                            # Initially empty, stores all generated embeddings:
│    ├── text_embeddings/            # Contains .npy files for text embeddings.
│    ├── augmented_text_embeddings/  # Contains .npy files for augmented text embeddings.
│    ├── frames_embedding/           # Contains .npy files for video frames embeddings.
│    ├── aug_frames_embeddings/      # Contains .npy files for augmented frames embeddings.
│    ├── mfcc_embeddings/            # Contains .npy files for MFCC features of audio.
│    ├── aug_mfcc_embeddings/        # Contains .npy files for augmented MFCC features.
```

### Data Folder Structure

The generated data and features are saved in various folders after preprocessing:

- `/data/mfcc_embeddings`: Contains MFCC features extracted from audio files.
- `/data/augAudio_wav`: Contains augmented audio files after translations.
- `/data/frames_each_audio`: Contains video frames extracted from audio-video pairs.
- `/data/frames_emd_each_audio`: Contains image embeddings of frames processed by CLIP.
- `/data/text_embeddings`: Contains sentence embeddings generated by `SentenceTransformer`.
- `/data/aug_frames_embeddings`: Contains augmented frames embeddings.

### Data Information

- **iemocap.csv**: Main dataset containing columns like `FileName`, `Sentences`, `Label`, `Text`, and `AugmentedText`.
- **iemocap_features.csv**: Contains extracted MFCC features, labels, and other details.

### Results

After training, the results will be saved in `result.txt` and include performance metrics on the IEMOCAP dataset.

## Approach

For each audio file, corresponding video frames are selected based on the dot product between the CLIP-generated text and image embeddings. The selected frames are then used in training, alongside the text and audio embeddings.

- **MFCC Features**: Audio is processed into MFCC features with dimensions `(20, num_frames)`.
- **CLIP Text and Image Embeddings**: Both text and frames are embedded into a 512-dimensional vector space using the CLIP model (`openai/clip-vit-base-patch32`).
- **Multitask Learning and Contrastive Learning**: The model learns unified embeddings using these techniques, optimizing the system for emotion classification.


### GPU and Model Usage

The scripts are designed to utilize GPU processing wherever available for faster processing of text and image embeddings.

---

## Contact

For any issues or queries, feel free to reach out via the GitHub Issues section.

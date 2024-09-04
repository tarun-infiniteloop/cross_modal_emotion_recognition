# Cross-Modal Emotion Recognition: A Unified Framework for Video, Audio, and Text Integration

This project presents a novel approach to multimodal emotion recognition, integrating video, audio, and text data. Our method generates a unified vector space using multitask learning, contrastive learning, and autoencoder techniques. The system outperforms current state-of-the-art methods on the IEMOCAP benchmark dataset.

![Alt Fig. 1. Illustraion of Cross-modal emotion recognition (CMER)](images/Fig1_4.png)

![Alt text](images/Fig2_2.png)

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Introduction
Emotion recognition is crucial for improving human-computer interaction. This project proposes a unified framework that integrates multiple modalities (video frames, audio, and text) to enhance emotion recognition accuracy.

### Key Features:
- Multimodal data fusion using Bi-directional LSTMs.
- Contrastive loss and reconstruction loss for robust feature learning.
- Outperforms state-of-the-art methods on the IEMOCAP dataset with 85.82% accuracy.

## Project Structure
```plaintext
├── images/                      # Contains figures used in the project
├── paper.tex                    # LaTeX file for the paper
├── refs.bib                     # Bibliography file
├── README.md                    # Project description
└── figures/                     # Directory for figures

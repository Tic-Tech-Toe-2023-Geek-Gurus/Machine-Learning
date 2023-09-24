# GEEK GURUS - Voice-Based Profile Unlock System 🎙️🔐

Welcome to the Geek Gurus' Voice-Based Profile Unlock System! In this project, we have developed a smart voice lock that can accurately identify and open a person's profile based on their voice. This system has been designed to provide secure access to individual user profiles across various applications, devices, and online platforms. Below, you'll find all the details you need to get started with our solution. 🚀

## Problem Statement:
Our challenge was to create a voice-based profile unlock system capable of distinguishing between different users based on their unique voice characteristics. We aimed to build a robust and reliable software solution that offers secure access control. 🎯

## Requirements:
To tackle this challenge, we followed a structured approach, including the following key components:

### Voice Data Collection:
- We collected a total of 18 precollected datasets.
- Additionally, we gathered 7 individual datasets to enhance our model's accuracy. 📊

### Voice Feature Extraction:
- We utilized the Short Time Fourier Transform (STFT) technique to extract voice features.
- These features were then converted into the dB scale.
- Finally, we generated a spectrogram for each sample to visualize the voice characteristics. 🎶

### Machine Learning Model:
- We trained a Convolutional Neural Network (CNN) to recognize voices accurately.
- Our model achieved an impressive accuracy rate of 92.58% in predicting results. 🤖

### User Profile Management:
- We implemented a simple user profile management system to associate voices with user profiles securely. 👤

## Results:
Our hard work and dedication resulted in the successful development of a CNN model for voice recognition with a remarkable accuracy rate of 92.58%. This model has proven to be highly effective in distinguishing between users based on their voice characteristics. 🎉

### Model Architecture:
To achieve this level of accuracy, we adopted a specific model architecture:

- Two Convolutional Layers: These layers increase computational efficiency by extracting essential features from the spectrogram images.
- Two Max Pooling Layers: These layers help in extracting dominant features critical for detecting frequency differences in users' voices.
- Two ReLU Layers: These introduce non-linearity into the model, accommodating complex functions that are essential for voice recognition. 🧠

## Getting Started:
To begin using our Voice-Based Profile Unlock System, please follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have the required dependencies installed (details provided in the documentation).
3. Collect voice data for both the precollected and individual datasets.
4. Extract voice features using the STFT and spectrogram generation process.
5. Train the CNN model using the provided code.
6. Implement user profile management to link voices with user profiles.
7. Test the system and enjoy the benefits of secure voice-based profile unlocking! 🚪🔊

## Conclusion:
We hope you find our Voice-Based Profile Unlock System both innovative and effective. It's been an exciting journey for the Geek Gurus team, and we believe our solution can revolutionize user authentication methods across various domains.

If you have any questions, suggestions, or feedback, please don't hesitate to reach out. We're here to help you make the most of our system.

Happy voice profiling! 🎤🔓
The Geek Gurus Team 🤓👩‍💻👨‍💻

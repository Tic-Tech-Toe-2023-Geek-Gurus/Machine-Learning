import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

def create_spectrograms(train_folder):
    images_folder = os.path.join(train_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    for root, _, files in os.walk(train_folder):
        if root != train_folder:  
            subfolder_name = os.path.basename(root)
            subfolder_images_folder = os.path.join(images_folder, subfolder_name)
            os.makedirs(subfolder_images_folder, exist_ok=True)

            for file in files:
                if file.endswith(".wav"):
                    wav_file_path = os.path.join(root, file)
                    x, sr = librosa.load(wav_file_path, sr=44100)
                    X = librosa.stft(x)
                    Xdb = librosa.amplitude_to_db(abs(X))
                    plt.figure(figsize=(10, 5))
                    ax = plt.axes()
                    ax.set_axis_off()
                    plt.set_cmap('hot')
                    librosa.display.specshow(Xdb, y_axis='log', x_axis='time', sr=sr)


                    output_path = os.path.join(subfolder_images_folder, os.path.splitext(file)[0] + ".png")
                    plt.savefig(output_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
                    plt.close()

if __name__ == "__main__":
    train_folder = r"C:\Users\Shashank\Documents\Tic-Tech-Toe-2023\temp"  
    create_spectrograms(train_folder)

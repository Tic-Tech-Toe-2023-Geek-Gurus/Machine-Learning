from pydub import AudioSegment
import os

input_audio_file = r"C:\Users\Shashank\Documents\Tic-Tech-Toe-2023\Test\audio\Himanshu\Himanshu.wav"
output_directory = r"C:\Users\Shashank\Documents\Tic-Tech-Toe-2023\temp\Himanshu"

os.makedirs(output_directory, exist_ok=True)

audio = AudioSegment.from_file(input_audio_file)

split_duration = 4 * 1000  # Convert to milliseconds

for i in range(30):
    start_time = i * split_duration
    end_time = (i + 1) * split_duration
    split_audio = audio[start_time:end_time]

    output_file = os.path.join(output_directory, f"output_audio_{i}.wav")
    split_audio.export(output_file, format="wav")

print("Audio file has been successfully split into 30 files.")

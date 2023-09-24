from pydub import AudioSegment
import os
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
import random
import tensorflow.keras.layers as tfl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle


class model:
        
    def create_spectrograms(self,input_folder, destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

        for root, _, files in os.walk(input_folder):
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
                    output_path = os.path.join(destination_folder, os.path.splitext(file)[0] + ".png")
                    plt.savefig(output_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
                    plt.close()

    def predict(self):
        filename = r"C:\Users\Shashank\Documents\Tic-Tech-Toe-2023\Backend_Django\api\ML-Model\conv_model.sav"
        with open(filename, 'rb') as file:
            trained_model = pickle.load(file)

        input_folder = r"C:\Users\Shashank\Documents\Tic-Tech-Toe-2023\Backend_Django\dataset"  
        destination_folder = r"C:\Users\Shashank\Documents\Tic-Tech-Toe-2023\Backend_Django\dataset\predict_images" 
        self.create_spectrograms(input_folder, destination_folder)
        
        image=Image.open(str(r"C:\Users\Shashank\Documents\Tic-Tech-Toe-2023\Backend_Django\dataset\predict_images\output_audio_1.png"))
        new_img=image.resize((200,200))
        NewEntryToCheck=np.array(new_img)/255.

        y_predicted = trained_model.predict(tf.expand_dims(NewEntryToCheck,axis=0))

        max_element = float('-inf')
        max_element_index = None

        for i, sublist in enumerate(y_predicted):
            for j, element in enumerate(sublist):
                if element > max_element:
                    max_element = element
                    max_element_index = (i, j)

        print(f"The maximum element is {max_element} at index {max_element_index}")
        result_key = ""
        my_dict = {}

        with open(r"C:\Users\Shashank\Documents\Tic-Tech-Toe-2023\Backend_Django\dataset\dictionary.txt", 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split(':')
                    key = key.strip().strip("'")  
                    value = int(value.strip())
                    my_dict[key] = value

        for key, value in my_dict.items():
            if value == max_element_index[1]:
                result_key = key
                break  
        
        print(result_key)

    def fitModel(self, input_audio_file):
        self.preprocess(input_audio_file,'./dataset/replicates','./dataset/replicates','./dataset/images')
        dataset_path="./dataset/images/"
        folders=os.listdir(dataset_path)

        X_train=[]
        y_train=[]
        X_test=[]
        y_test=[]

        num=np.random.rand(3960)
        mask=num<0.2
        split=mask.astype(int)

        i=0
        for dirs in folders:
            for img in os.listdir(str(dataset_path+dirs)):
                image=Image.open(str(dataset_path+dirs+'/'+img))
                new_img=image.resize((200,200))
                tmp_array=np.array(new_img)/255.
                if split[i]==0:
                    X_train.append(tmp_array)
                    y_train.append(str(dirs))
                else:
                    X_test.append(tmp_array)
                    y_test.append(str(dirs))
                i=i+1
        
        dict={}
        i=0
        for val in folders:
            dict[val]=i
            i=i+1

        i=0
        for val in y_train:
            y_train[i]=dict[y_train[i]]
            i=i+1

        i=0
        for val in y_test:
            y_test[i]=dict[y_test[i]]
            i=i+1

        conv_model = self.convolutional_model((200, 200, 4),len(dict))
        conv_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
        history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)

        conv_model.save("./ML-Model/my-model")

    def replicate_files(folder_path, num_replications=15):
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                for i in range(num_replications):
                    new_filename = f"{os.path.splitext(filename)[0]}_replicate_{i}{os.path.splitext(filename)[1]}"
                    new_file_path = os.path.join(folder_path, new_filename)
                    shutil.copy2(file_path, new_file_path)
                    print(f"Replicated '{filename}' as '{new_filename}'")

    def convolutional_model(input_shape,size):
            input_img = tf.keras.Input(shape=input_shape)
            Z1=tfl.Conv2D(filters=8,kernel_size=(4,4),strides=(1,1),padding='same')(input_img)
            A1=tfl.ReLU()(Z1)
            P1=tfl.MaxPool2D(pool_size=(8,8),strides=(8,8),padding='same')(A1)
            Z2=tfl.Conv2D(filters=16,kernel_size=(2,2),strides=(1,1),padding='same')(P1)
            A2=tfl.ReLU()(Z2)
            P2=tfl.MaxPool2D(pool_size=(4,4),strides=(4,4),padding='same')(A2)
            F=tfl.Flatten()(P2)

            model = tf.keras.Model(inputs=input_img, outputs=outputs)
            return model

    def preprocess(self, input_audio_file,folder_path,input_folder,destination_folder):
        output_directory = './dataset/replicates'
        audio = AudioSegment.from_file(input_audio_file)

        split_duration = 4 * 1000 

        for i in range(30):
            start_time = i * split_duration
            end_time = (i + 1) * split_duration
            split_audio = audio[start_time:end_time]
            output_file = os.path.join(output_directory, f"output_audio_{i}.wav")
            split_audio.export(output_file, format="wav")

        folder_path = './dataset/replicates'
        self.replicate_files(folder_path)
        input_folder = r"./dataset/replicates"
        destination_folder = r"./dataset/images" 
        self.create_spectrograms(input_folder, destination_folder)
        
obj=model()

obj.predict()

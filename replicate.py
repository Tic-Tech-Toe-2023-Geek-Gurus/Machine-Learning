import os
import shutil

def replicate_files(folder_path, num_replications=5):
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

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    replicate_files(folder_path)

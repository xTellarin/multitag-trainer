import os
from dotenv import load_dotenv

load_dotenv()
training_folder = os.getenv('TRAINING')
folder_path = training_folder

tag = 'YOUR-TAG(S)'

for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        text_filename = file_name+'.txt'

        with open(os.path.join(folder_path, text_filename), 'w') as f:
            f.write(tag)
import os
import io

# folder path
dir_path = r'Path_to_the_Input_images_folder'

file_name = './Destination_path/name_of_file.txt'
# example
# file_name = 'D_drive/name_of_file.txt

content = []

for path in os.scandir(dir_path):
    if path.is_file():
        with io.open(file_name, 'w') as file:
            content.append(path.name)
            listToStr = ' \n'.join([str(elem) for i, elem in enumerate(content)])
            file.write(listToStr)


import json
import os


folder_path = 'dataset'

#trich raw code tu file json

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path,'r') as file:
        data_list = json.load(file)
        if filename == 'non-vulnerables.json':
            for index,data in enumerate(data_list):
                code_file_name = f'0_{index}.c'
                output_path = f'output/code/non-vulnerables/{index}'
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                code_file_path = os.path.join(output_path,code_file_name)
                with open(code_file_path, 'w',encoding='utf-8') as file:
                    file.write(data.get('code'))  
        elif filename == 'vulnerables.json':
            for index,data in enumerate(data_list):
                code_file_name = f'1_{index}.c'
                output_path = f'output/code/vulnerables/{index}'
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                code_file_path = os.path.join(output_path,code_file_name)
                with open(code_file_path, 'w',encoding='utf-8') as file:
                    file.write(data.get('code'))  





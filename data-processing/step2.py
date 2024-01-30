import os

#dua raw code qua joern

current_directory = os.path.dirname(os.path.abspath(__file__))
list_code_path = 'output/code/non-vulnerables'

def parse_code(input_path, output_path):
    joern_directory = r'code-slicer\joern\y\joern-cli'
    os.chdir(joern_directory)
    joern_parse_command = f'joern-parse {current_directory}\{input_path}'
    joern_export_command = f'joern-export --repr cpg14 --out {current_directory}\{output_path}'
    os.system(joern_parse_command)
    os.system(joern_export_command)



# Sử dụng os.listdir để lấy danh sách các tên thư mục
subdirectories = [f for f in os.listdir(
    list_code_path) if os.path.isdir(os.path.join(list_code_path, f))]

for subdirectory in subdirectories:
    os.chdir(current_directory)
    code_path = list_code_path + f'/1813'
    parsed_path = f'output/joern-output/non-vulnerables/1813'
    if os.path.exists(parsed_path):
        continue
    parse_code(code_path, parsed_path)

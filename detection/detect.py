import os
import torch
import shutil
import subprocess
import helper.parse.parse as parse
import helper.parse.graph as graph
import helper.ggnn.ggnn as ggnn
import helper.parse.data_entry as g


#cho vao 1 string hoac 1 file source code ket qua dau ra la 1 neu la vulnerable hoac 0 neu khong la vulnerable

storage_dir = "./storage"
current_directory = os.path.dirname(os.path.abspath(__file__))

code = """static int jbig2_word_stream_buf_get_next_word ( Jbig2WordStream * self , int offset , uint32_t * word ) {\n Jbig2WordStreamBuf * z = ( Jbig2WordStreamBuf * ) self ;\n const byte * data = z -> data ;\n uint32_t result ;\n if ( offset + 4 < z -> size ) result = ( data [ offset ] << 24 ) | ( data [ offset + 1 ] << 16 ) | ( data [ offset + 2 ] << 8 ) | data [ offset + 3 ] ;\n else if ( offset > z -> size ) return - 1 ;\n else {\n int i ;\n result = 0 ;\n for ( i = 0 ;\n i < z -> size - offset ;\n i ++ ) result |= data [ offset + i ] << ( ( 3 - i ) << 3 ) ;\n }\n * word = result ;\n return 0 ;\n }"""

def parse_code(input_path, output_path):
    joern_directory = r'helper\joern\joern-cli'
    if not os.path.exists(joern_directory):
       os.chdir(r'helper\joern')
       subprocess.call(['bash', 'get_joern.sh'])
       os.chdir(current_directory)
    os.chdir(joern_directory)
    joern_parse_command = f'joern-parse {current_directory}\{input_path}'
    joern_export_command = f'joern-export --repr cpg14 --out {current_directory}\{output_path}'
    os.system(joern_parse_command)
    os.system(joern_export_command)


def create_joern_input(code):
    file_path = storage_dir + "/code.c"
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(code)
    if os.path.exists(file_path):
        return file_path
    return


def get_nodes_and_edges(joern_output):
    for dot_file_path in os.listdir(joern_output):
        nodes, edges = parse.read_dot_file(
            os.path.join(joern_output, dot_file_path))
        parse.write_to_csv(nodes, edges, storage_dir)


def run_joern(code=None, file_path=None):
    joern_output_path = storage_dir + "/joern_output"
    if code:
        file_path = create_joern_input(code)
    if file_path:
        parse_code(file_path, joern_output_path)
        os.chdir(current_directory)
        if code:
            os.remove(file_path)
    if os.path.exists(joern_output_path):
        get_nodes_and_edges(joern_output_path)
        shutil.rmtree(joern_output_path)


def detect(code=None,file_path=None):
    run_joern(code=code,file_path=file_path)
    _graph = graph.get_graph(storage_dir)
    model = ggnn.GGNNSum(input_dim=100, output_dim=200,
                         max_edge_types=g.max_etype)
    model.load_state_dict(torch.load(
        "./helper/ggnn/GGNNSumModel-model.bin", map_location=torch.device('cpu')))
    prediction = model(_graph)
    threshold = 0.5
    predicted_label = 1 if prediction.item() > threshold else 0
    return predicted_label

if __name__ == "__main__":
    print(detect("X42.c"))
    print(detect(code))

import os
import sys
import torch
import shutil
import subprocess
import helper.parse.parse as parse
import helper.parse.graph as graph
import helper.ggnn.ggnn as ggnn
import helper.parse.data_entry as g

import json
import argparse

# cho vao 1 string hoac 1 file source code ket qua dau ra la 1 neu la vulnerable hoac 0 neu khong la vulnerable

storage_dir = "./storage"
current_directory = os.path.dirname(os.path.abspath(__file__))


def parse_code(input_path, output_path):
    joern_directory = r'helper/joern/joern-cli'
    if not os.path.exists(joern_directory):
        os.chdir(r'helper/joern')
        subprocess.call(['bash', 'get_joern.sh'])
        os.chdir(current_directory)

    # debug
    print(joern_directory)

    os.chdir(joern_directory)
    joern_parse_command = f'./joern-parse {current_directory}/{input_path}'
    joern_export_command = f'./joern-export --repr cpg14 --out {current_directory}/{output_path}'

    # debug
    print(joern_parse_command)
    print(joern_export_command)

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


def detect(code=None, file_path=None):

    run_joern(code=code, file_path=file_path)
    _graph = graph.get_graph(storage_dir)
    model = ggnn.GGNNSum(input_dim=100, output_dim=200,
                         max_edge_types=g.max_etype)
    model.load_state_dict(torch.load(
        "./helper/ggnn/GGNNSumModel-model.bin", map_location=torch.device('cpu')))
    prediction = model(_graph)
    threshold = 0.5
    predicted_label = 1 if prediction.item() > threshold else 0
    return predicted_label


code_non_vul = """static int jbig2_word_stream_buf_get_next_word ( Jbig2WordStream * self , int offset , uint32_t * word ) {\n Jbig2WordStreamBuf * z = ( Jbig2WordStreamBuf * ) self ;\n const byte * data = z -> data ;\n uint32_t result ;\n if ( offset + 4 < z -> size ) result = ( data [ offset ] << 24 ) | ( data [ offset + 1 ] << 16 ) | ( data [ offset + 2 ] << 8 ) | data [ offset + 3 ] ;\n else if ( offset > z -> size ) return - 1 ;\n else {\n int i ;\n result = 0 ;\n for ( i = 0 ;\n i < z -> size - offset ;\n i ++ ) result |= data [ offset + i ] << ( ( 3 - i ) << 3 ) ;\n }\n * word = result ;\n return 0 ;\n }"""

code_vul = """static int alloc_addbyter ( int output , FILE * data ) {\n struct asprintf * infop = ( struct asprintf * ) data ;\n unsigned char outc = ( unsigned char ) output ;\n if ( ! infop -> buffer ) {\n infop -> buffer = malloc ( 32 ) ;\n if ( ! infop -> buffer ) {\n infop -> fail = 1 ;\n return - 1 ;\n }\n infop -> alloc = 32 ;\n infop -> len = 0 ;\n }\n else if ( infop -> len + 1 >= infop -> alloc ) {\n char * newptr ;\n newptr = realloc ( infop -> buffer , infop -> alloc * 2 ) ;\n if ( ! newptr ) {\n infop -> fail = 1 ;\n return - 1 ;\n }\n infop -> buffer = newptr ;\n infop -> alloc *= 2 ;\n }\n infop -> buffer [ infop -> len ] = outc ;\n infop -> len ++ ;\n return outc ;\n }"""

# Convention, leave everything outside this block untouched
if __name__ == "__main__":

    # extract input
    input_dict = json.loads(sys.argv[-1])

    # debug
    print(f'Current working directory:{os.getcwd()}')
    print(f'inputDict: {input_dict}')

    # main logic
    is_vulnerable = detect(input_dict['unprocessedContentFunc'])

    # keep in mind, boolean is different from number(0, 1)
    output_dict = {
        "modelName": 'devign',
        "isVulnerable": True if is_vulnerable == 1 else False
    }
    output_json = json.dumps(output_dict)

    print(output_json)

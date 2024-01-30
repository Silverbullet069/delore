import os
import csv
from gensim.models import FastText
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import json

# tao input cua ggnn

cpg_path = 'output/cpg/non-vulnerables'
subdirectories = [f for f in os.listdir(
    cpg_path) if os.path.isdir(os.path.join(cpg_path, f))]
json_output_path = 'output/ggnn_input/GGNN_input_3.json'

# node extraction
type_list = [
    "METHOD", "BLOCK", "PARAM", "LOCAL", "IDENTIFIER",
    "COMMENT", "UNKNOWN", "CONTROL_STRUCTURE", "FIELD_IDENTIFIER",
    "FILE", "LITERAL", "RETURN", "METHOD_RETURN", "TYPE_DECL",
    "MEMBER", "META_DATA", "METHOD_REF", "NAMESPACE_BLOCK",
    "NAMESPACE", "CALL"
]
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(np.array(type_list).reshape(-1, 1))

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}'
}


def train_fasttext_model(tokens, vector_length):
    print('\ntraining w2v model...')
    # Set min_count to 1 to prevent out-of-vocabulary errors
    model = FastText(tokens, min_count=1, vector_size=vector_length, sg=1)
    embeddings = model.wv
    del model
    return embeddings


def tokenize(line):
    tmp, w = [], []
    i = 0
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i+3] in operators3:
            tmp.append(''.join(w))
            tmp.append(line[i:i+3])
            w = []
            i += 3
        elif line[i:i+2] in operators2:
            tmp.append(''.join(w))
            tmp.append(line[i:i+2])
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w))
            tmp.append(line[i])
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    # Add the last word to the result list
    tmp.append(''.join(w))
    # Filter out irrelevant strings
    res = list(filter(lambda c: c != '', tmp))
    return list(filter(lambda c: c != ' ', res))


def read_csv(csv_file_path):
    data = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as readfile:
        reader = csv.DictReader(readfile)
        for row in reader:
            data.append(dict(row))
    return data


def vectorize(tokenized, vector_length, embeddings):
    vectors = np.zeros(shape=(1, vector_length))
    for i in range(min(len(tokenized), vector_length)):
        vectors[0][i] = embeddings[tokenized[i]][0]
    return vectors[0]


tokens = []
for index, subdirectory in enumerate(subdirectories):
    print(f'tokens collecting....{index}/{len(subdirectories)}', end='\r')
    nodes_csv_path = cpg_path + '/' + subdirectory + '/' + 'nodes.csv'
    nodes = read_csv(nodes_csv_path)
    for node in nodes:
        code = node['code'].replace(" ", "")
        if ',' in code:
            if code.split(',')[0] in code.split(',')[1] or code.split(',')[0].lower() in code.split(',')[1]:
                code = code.split(',')[1]
        tokenized = tokenize(code)
        for token in tokenized:
            tokens.append(token)
embeddings = train_fasttext_model(tokens, 1)


def create_node_features(nodes):
    node_features = []
    for node in nodes:
        code = node['code'].replace(" ", "")
        if ',' in code:
            if code.split(',')[0] in code.split(',')[1] or code.split(',')[0].lower() in code.split(',')[1]:
                code = code.split(',')[1]
        tokenized = tokenize(code)
        vectorized = vectorize(tokenized, 80, embeddings)

        type = node['type'].split(',')[0]
        if type not in type_list:
            type = 'CALL'
        type_encode = one_hot_encoded[type_list.index(type)]
        graph_feature = np.concatenate((vectorized, type_encode))
        node_features.append(graph_feature)
    return node_features


data = []


def get_data(file_path):
    count = 0
    subdirectories = [f for f in os.listdir(
            file_path) if os.path.isdir(os.path.join(file_path, f))]
    for subdirectory in subdirectories:
       
            print(f'proccessing: {count}/{len(subdirectories)} - at {subdirectory}', end='\r')
            nodes_csv_path = cpg_path + '/' + subdirectory + '/' + 'nodes.csv'
            edges_csv_path = cpg_path + '/' + subdirectory + '/' + 'edges.csv'
            nodes = read_csv(nodes_csv_path)
            edges = read_csv(edges_csv_path)

            node_features = create_node_features(nodes)
            node_features_list = [arr.tolist() for arr in node_features]

            graph = []
            for edge in edges:
                edge_type = edge['type']
                source = int(edge['start']) - 5
                destination = int(edge['end']) - 5
                tuple =[
                   source,
                    edge_type,
                    destination
                ]
                graph.append(tuple)

            data_dict = {
                'node_features': node_features_list,
                'graph': graph,
                'target': 0
            }
            with open(json_output_path, 'a') as json_file:
            # Ghi dấu phẩy để ngăn cách giữa các đối tượng JSON nếu có nhiều đối tượng
                if json_file.tell() != 0:
                    json_file.write(',')

                # Ghi dữ liệu vào tệp JSON
                json.dump(data_dict, json_file)
                if(count == len(subdirectories)-1):
                    json_file.write("]")
            count += 1
get_data('output/cpg/non-vulnerables')

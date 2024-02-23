from sklearn.preprocessing import OneHotEncoder
import numpy as np
import csv
from gensim.models import FastText
import helper.parse.data_entry as g


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

model = FastText.load("./helper/parse/fasttext_model.bin")
# embeddings = KeyedVectors.load_word2vec_format("embeddings.vec")


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


def vectorize(tokenized, vector_length):
    vectors = np.zeros(shape=(1, vector_length))
    for i in range(min(len(tokenized), vector_length)):
        vectors[0][i] = model.wv[i]
    return vectors[0]


def create_node_features(nodes):
    node_features = []
    for node in nodes:
        code = node['code'].replace(" ", "")
        if ',' in code:
            if code.split(',')[0] in code.split(',')[1] or code.split(',')[0].lower() in code.split(',')[1]:
                code = code.split(',')[1]
        tokenized = tokenize(code)
        vectorized = vectorize(tokenized, 80)

        type = node['type'].split(',')[0]
        if type not in type_list:
            type = 'CALL'
        type_encode = one_hot_encoded[type_list.index(type)]
        graph_feature = np.concatenate((vectorized, type_encode))
        node_features.append(graph_feature)
    return node_features


def get_graph(path):
    nodes_csv_path = path + '/' + 'nodes.csv'
    edges_csv_path = path + '/' + 'edges.csv'
    nodes = read_csv(nodes_csv_path)
    edges = read_csv(edges_csv_path)
    node_features = create_node_features(nodes)
    graph = []
    for edge in edges:
        edge_type = edge['type']
        source = int(edge['start']) - 5
        destination = int(edge['end']) - 5
        tuple = [
            source,
            edge_type,
            destination
        ]
        graph.append(tuple)
    graph_entry = g.DataEntry(len(node_features), node_features, graph)
    return graph_entry.graph

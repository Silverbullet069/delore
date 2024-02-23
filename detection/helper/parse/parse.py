import os
import pydot
import csv
import re
from html import unescape


def parse_label(label):
    # Updated regex to allow optional spaces
    type_parttern = re.compile(r'\(([^<>]+),')
    code_parttern = re.compile(r',([^<>]+)\)')
    location_pattern = re.compile(r'<SUB>([^<>]+)</SUB>')

    # Search for the pattern in the label
    type_match = type_parttern.search(label)
    code_match = code_parttern.search(label)
    location_match = location_pattern.search(label)

    if type_match or code_match or location_match:
        # Extract components from the match
        if type_match.group(1) is not None:
            type = type_match.group(1) if type_match else ""
            type = unescape(type)
        if code_match.group(1) is not None:
            code = code_match.group(1) if code_match else ""
            code = unescape(code)
        location = location_match.group(1) if location_match else ""
        return type, code, location
    else:
        return None


def read_dot_file(file_path):
    # Đọc tệp tin DOT và tạo đồ thị
    graph = pydot.graph_from_dot_file(file_path)[0]

    # Lấy danh sách các node và cạnh
    nodes = graph.get_nodes()
    edges = graph.get_edges()
    return nodes, edges


def write_to_csv(nodes, edges, csv_file_path):
    try:
        nodes_file = csv_file_path + '/nodes.csv'
        edges_file = csv_file_path + '/edges.csv'

        node_fieldnames = ['key', 'location', 'code', 'type']
        if not os.path.exists(nodes_file):
            with open(nodes_file, 'w', newline='') as csvfile:
                csv_writer = csv.DictWriter(
                    csvfile, fieldnames=node_fieldnames)
                csv_writer.writeheader()

        with open(nodes_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=node_fieldnames)
            # Ghi thông tin vào tệp CSV, tránh các node trùng lặp
            existing_nodes = set()
            try:
                with open(nodes_file, 'r', newline='', encoding='utf-8') as readfile:
                    reader = csv.DictReader(readfile)
                    for row in reader:
                        existing_nodes.add(row['key'])
            except FileNotFoundError:
                pass

            for node in nodes:
                key = node.get_name().strip('"')
                if key not in existing_nodes:
                    type, code, location = parse_label(node.get_label())
                    writer.writerow({
                        'key': key,
                        'location': location,
                        'code': code,
                        'type': type
                    })
                    existing_nodes.add(key)

        # print(f"Data written to {nodes_file}")

        edge_fieldnames = ['start', 'end', 'type']
        if not os.path.exists(edges_file):
            with open(edges_file, 'w', newline='') as csvfile:
                csv_writer = csv.DictWriter(
                    csvfile, fieldnames=edge_fieldnames)
                csv_writer.writeheader()

        with open(edges_file, 'a', newline='', encoding='utf-8') as csvfile:
            existing_edges = set()
            try:
                with open(edges_file, 'r', newline='', encoding='utf-8') as readfile:
                    reader = csv.DictReader(readfile)
                    for row in reader:
                        start = row['start']
                        end = row['end']
                        edge = (start, end)
                        existing_edges.add(edge)
            except FileNotFoundError:
                pass

            writer = csv.DictWriter(csvfile, fieldnames=edge_fieldnames)
            for edge in edges:
                start = edge.get_source().strip('"')
                end = edge.get_destination().strip('"')
                if (start, end) not in existing_edges:
                    label = edge.get_label()
                    type = label.strip('"').split(':')[0]
                    writer.writerow({
                        'start': start,
                        'end': end,
                        'type': type
                    })
                existing_edges.add((start,end))


        # print(f"Data written to {edges_file}")
    except Exception as e:
        print(f"Error reading DOT file: {e}")

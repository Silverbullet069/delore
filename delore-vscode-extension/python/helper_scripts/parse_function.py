
import sys
import os.path
import clang.cindex
import re
import argparse
import json

TWO_FUNCTIONS = """
int add(int a, int b) {
    return a + b;
}

char toUpperCase(char c) {
    if ('A' <= c && c <= 'Z') {
        return c;
    } else if ('a' <= c && c <= 'z')
        return c - ('a' - 'A');
    else
        throw("[ERROR] toUpperCase(char c): c not an alphabet!");
}

void test() {
    return;
}
"""


def parse_function(code, is_context_ignored=True):

    # Initialize clang index
    idx = clang.cindex.Index.create()

    # Parse the code snippet
    try:
        tu = idx.parse('tmp.cpp', args=['-std=c++11'],
                       unsaved_files=[('tmp.cpp', code)])
    except clang.cindex.TranslationUnitLoadError as e:
        # Exception are handled on the other side
        print(str(e), file=sys.stderr)
        exit(1)

    # Check if any syntax errors occurred
    if not is_context_ignored and tu.diagnostics:
        # Exception are handled on the other side
        print("\n".join(str(d) for d in tu.diagnostics), file=sys.stderr)
        exit(2)

    # Extract function
    funcs = []
    root_cursor = tu.cursor
    for node in root_cursor.get_children():
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            func_start_offset = node.extent.start.offset
            func_end_offset = node.extent.end.offset
            func = {
                "name": node.spelling,
                "body": code[func_start_offset:func_end_offset]
            }
            funcs.append(func)

            # Debug
            # print(node.spelling)
            # print(node.extent.start.offset)
            # print(node.extent.end.offset)
            # print(code[node.extent.start.offset:node.extent.end.offset])
            # print('-----------------------------')

    return json.dumps({
        "msg": "This is a debug message from Python",
        "data": funcs
    })


if __name__ == '__main__':
    # print(parse_function(TWO_FUNCTIONS))

    program_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser(program_name)
    parser.add_argument(
        "code", help="C++ code snippet that needs to be checked", type=str)
    args = parser.parse_args()
    print(parse_function(args.code))

    pass

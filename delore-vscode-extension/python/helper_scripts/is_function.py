import sys
import re
import json
import clang.cindex
import os.path
import argparse


def is_function(code, is_context_ignored=True):

    # Initialize clang index
    idx = clang.cindex.Index.create()

    # Parse the code snippet
    try:
        tu = idx.parse('tmp.cpp', unsaved_files=[('tmp.cpp', code)])
    except clang.cindex.TranslationUnitLoadError as e:
        # Exception are handled on the other side
        print(str(e), file=sys.stderr)
        exit(1)

    # Check if any syntax errors occurred
    if not is_context_ignored and tu.diagnostics:
        # Exception are handled on the other side
        print("\n".join(str(d) for d in tu.diagnostics), file=sys.stderr)
        exit(2)

    # Traverse the AST
    for node in tu.cursor.walk_preorder():
        # print(node.kind)
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            return json.dumps({
                "msg": "This is a C++ function!",
                "data": True,
            })

    # This is converted into a string when called inside TypeScript runtime
    return json.dumps({
        "msg": "This is NOT a C++ function!",
        "data": False,
    })


# debug
CPP_CODE_SNIPPET = """
void tmp(int a, int b) {
    return;
}
"""

CPP_CODE_SNIPPET_2 = """
int add(int a, int b) {
    return a + b;
}
"""

if __name__ == '__main__':
    # print(is_function(CPP_CODE_SNIPPET))
    # print(is_function(CPP_CODE_SNIPPET_2))

    parser = argparse.ArgumentParser(os.path.basename(__file__))
    parser.add_argument(
        "code", help="A C/C++ function that needs to be checked", type=str)
    args = parser.parse_args()

    print(is_function(args.code))

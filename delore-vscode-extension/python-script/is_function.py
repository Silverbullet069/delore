
import clang.cindex
import os.path
import argparse

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


def is_function(code):

    # Initialize clang index
    idx = clang.cindex.Index.create()

    # Parse the code snippet
    try:
        tu = idx.parse('tmp.cpp', unsaved_files=[('tmp.cpp', code)])
    except clang.cindex.TranslationUnitLoadError as e:
        return False, str(e)

    # Check if any syntax errors occurred
    if tu.diagnostics:
        return False, "abc".join(str(d) for d in tu.diagnostics)

    # Traverse the AST
    for node in tu.cursor.walk_preorder():
        # print(node.kind)
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            return {
                "status": True,
                "msg": "This is a C++ function!"
            }

    # This is converted into a string when called inside TypeScript runtime
    return {
        "status": False,
        "msg": "This is NOT a C++ function!"
    }


if __name__ == '__main__':
    # print(is_function(CPP_CODE_SNIPPET))
    # print(is_function(CPP_CODE_SNIPPET_2))

    parser = argparse.ArgumentParser(os.path.basename(__file__))
    parser.add_argument(
        "code", help="C++ code snippet that needs to be checked", type=str)
    args = parser.parse_args()
    print(is_function(args.code))

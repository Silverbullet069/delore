import argparse
import json
import helper.feat_extraction.feat_ext as feat_ext
import os
import helper.joern.joern as joern
import dgl
import helper.codeBert.codebert as cb
import torch as th
import pandas as pd
import helper.model.LitGNN as model
import re

current_directory = os.path.dirname(os.path.abspath(__file__))
model = model.LitGNN.load_from_checkpoint(
    "storage/checkpoint/epoch=3-step=13849.ckpt")


def update_line(lines, number, new_isVul):
    for line in lines:
        if line['num'] == number:
            line['isVulnerable'] = new_isVul
            return line
    return None


def detect(code):
    path = joern.create_joern_input(code)
    model.eval()
    graphs = feat_ext.processing(
        path, graph_type="pdg+raw", current_directory=current_directory)
    for g in graphs:
        g = model(g)
        h_func = g.ndata['pred_func']

        # debug
        print(h_func[0][1])

        # Threshold = 0.75
        if h_func[0][1] > 0.75:
            return True
    return False


def localization(lines):
    code = "\n".join(lines)
    path = joern.create_joern_input(code)

    result = []
    for i in range(0, len(lines)):
        line = {
            "num": i,
            "score": 0,
            "isVulnerable": False,
            "content": lines[i]
        }
        result.append(line)

    model.eval()

    graphs = feat_ext.processing(
        path, graph_type="pdg+raw", current_directory=current_directory)

    for g in graphs:
        g = model(g)
        h = g.ndata['pred']
        k = min(h.size(0), 10)  # handle case length < 10
        top_10 = th.topk(h[:, 1], k=k).indices
        line = g.ndata["_LINE"]
        for i in top_10:
            res = update_line(result, line[i].item()-1, True)
    return result


mock_lines = [
    "PHP_FUNCTION(imageconvolution)",
    "{",
    "    zval *SIM, *hash_matrix;",
    "    zval **var = NULL, **var2 = NULL;",
    "    gdImagePtr im_src = NULL;",
    "    double div, offset;",
    "    int nelem, i, j, res;",
    "    float matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};",
    "",
    "    if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, \"radd\", &SIM, &hash_matrix, &div, &offset) == FAILURE) {",
    "        RETURN_FALSE;",
    "    }",
    "",
    "    ZEND_FETCH_RESOURCE(im_src, gdImagePtr, &SIM, -1, \"Image\", le_gd);",
    "",
    "    nelem = zend_hash_num_elements(Z_ARRVAL_P(hash_matrix));",
    "    if (nelem != 3) {",
    "        php_error_docref(NULL TSRMLS_CC, E_WARNING, \"You must have 3x3 array\");",
    "        RETURN_FALSE;",
    "    }",
    "",
    "    for (i = 0; i < 3; i++) {",
    "        if (zend_hash_index_find(Z_ARRVAL_P(hash_matrix), (i), (void **)&var) == SUCCESS && Z_TYPE_PP(var) == IS_ARRAY) {",
    "            if (Z_TYPE_PP(var) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_PP(var)) != 3) {",
    "                php_error_docref(NULL TSRMLS_CC, E_WARNING, \"You must have 3x3 array\");",
    "                RETURN_FALSE;",
    "            }",
    "",
    "            for (j = 0; j < 3; j++) {",
    "                if (zend_hash_index_find(Z_ARRVAL_PP(var), (j), (void **)&var2) == SUCCESS) {",
    "                    SEPARATE_ZVAL(var2);",
    "                    convert_to_double(*var2);",
    "                    matrix[i][j] = (float)Z_DVAL_PP(var2);",
    "                } else {",
    "                    php_error_docref(NULL TSRMLS_CC, E_WARNING, \"You must have a 3x3 matrix\");",
    "                    RETURN_FALSE;",
    "                }",
    "            }",
    "        }",
    "    }",
    "    res = gdImageConvolution(im_src, matrix, (float)div, (float)offset);",
    "",
    "    if (res) {",
    "        RETURN_TRUE;",
    "    } else {",
    "        RETURN_FALSE;",
    "    }",
    "}"
]


def main():
    parser = argparse.ArgumentParser()

    # new parameters
    parser.add_argument("--function_level", action='store_true',
                        help="Whether to run model to only detect vulnerability at function-level")
    parser.add_argument("--line_level", action='store_true',
                        help="Whether to run model to only detect vulnerability at line-level")
    parser.add_argument("--input_json", default=None, type=str, required=True,
                        help='The JSON string that contains the function content.')

    args = parser.parse_args()

    standard_input = json.loads(args.input_json)

    # debug
    print(f'Current working directory: {os.getcwd()}')
    print(f'inputDict: {standard_input}')

    input_model_name = standard_input["modelName"]
    input_lines = standard_input["lines"]

    # function
    if args.function_level:
        # Preserve tabs and spaces
        code = "\n".join(input_lines)
        res = detect(code)

        standard_output = json.dumps({
            "modelName": input_model_name,
            "isVulnerable": res
        })
        print(standard_output)
        return

    # line
    if args.line_level:
        # Preserce tabs and spaces
        res = localization(input_lines)

        standard_output = json.dumps({
            "modelName": input_model_name,
            "lines": res
        })
        print(standard_output)
        return


if __name__ == "__main__":
    main()

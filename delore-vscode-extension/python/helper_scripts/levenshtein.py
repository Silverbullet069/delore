
import argparse
import os


def levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    matrix = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]

    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(
                    matrix[i - 1][j - 1] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j] + 1
                )

    return matrix[-1][-1]


def are_strings_similar(a, b, similarity_threshold=0.8):
    distance = levenshtein(a, b)
    longest_length = max(len(a), len(b))
    similarity = (longest_length - distance) / longest_length
    return similarity >= similarity_threshold


if __name__ == '__main__':

    parser = argparse.ArgumentParser(os.path.basename(__file__))
    parser.add_argument(
        "str1", help="First string", type=str
    )
    parser.add_argument(
        "str2", help="Second string", type=str
    )
    args = parser.parse_args()

    print(are_strings_similar(args.str1, args.str2))
    pass

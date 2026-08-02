from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader import load_matlab_file

MATLAB_LANG = Language(language())


def parse_and_print_tree(path):
    source = load_matlab_file(path)
    parser = Parser(MATLAB_LANG)
    tree = parser.parse(bytes(source, "utf-8"))
    print(tree.root_node)


if __name__ == "__main__":
    parse_and_print_tree("/sample_matlab/fft_basic.m")

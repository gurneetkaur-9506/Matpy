from pprint import pprint

from tree_sitter import Language, Parser
from tree_sitter_matlab import language

from reader import build_structure, load_matlab_file, structure_to_dict

SAMPLES = [
    "/sample_matlab/fft_basic.m",
    "/sample_matlab/indexing_ops.m",
    "/sample_matlab/beamform_basic.m",
    "/sample_matlab/builtins_demo.m",
]


def main():
    parser = Parser(Language(language()))
    for path in SAMPLES:
        tree = parser.parse(load_matlab_file(path).encode("utf-8"))
        print("=" * 25, path, "=" * 25)
        pprint(structure_to_dict(build_structure(tree)))
        print()


if __name__ == "__main__":
    main()

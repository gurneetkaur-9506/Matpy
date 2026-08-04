import sys

from translator import translate_file

MATLAB_FILE = "/sample_matlab/indexing_ops.m"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else MATLAB_FILE
    result = translate_file(path)
    print(result["python"], end="")


if __name__ == "__main__":
    main()

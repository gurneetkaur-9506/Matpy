import ast


def load_python_file(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    print(ast.dump(tree, indent=2, include_attributes=True))
    return tree


if __name__ == "__main__":
    from repo_paths import sample_python

    load_python_file(sample_python("fft_basic_py.py"))

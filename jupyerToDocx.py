import os
import sys
import re
import subprocess


def jupyter_to_markdown(path):
    f"""jupyter nbconvert --to markdown titanic_transform.ipynb"""

    args = ['jupyter', 'nbconvert', '--to', 'html', path]
    child = subprocess.call(args)


def markdown_to_docx(path):

    """pandoc -f titanic_transform.md -t html | pandoc -f html -t docx -o titanic.docx"""

    """pandoc titanic_transform.html -s -o titanic.docx"""

    """titanic_transform.html  -f markdown -t html | pandoc -f html -t docx -o titanic.docx"""
    args = ['pandoc', path, '-s', '-o', './jupyter/titanic.docx']
    child = subprocess.call(args)



def mark_down_del_python_code(path, out_path):
    with open(path, 'r', encoding='UTF8') as f:
        lines = []
        for line in f.readlines():
            lines += [line]

    oneline = "".join(lines)
    # print(oneline)

    r = re.compile("```python[^`]*```", re.DOTALL)

    for i in list(r.findall(oneline)):
        oneline = oneline.replace(i, "")

        # print(i)

    # print(type(oneline))
    with open(path, 'w', encoding='UTF8') as f:
        f.write(str(oneline))


if __name__ == '__main__':
    path = './jupyter/titanic_transform.ipynb'
    jupyter_to_markdown(path)

    md_path = './jupyter/titanic_transform.md'
    mark_down_del_python_code(md_path, None)




    # markdown_to_docx(md_path)

    pass

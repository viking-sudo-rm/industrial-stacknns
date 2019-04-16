import os
import sys


RAW_DIRNAME = "../data/treebank/raw"

total_count = 0
for filename in sorted(os.listdir(RAW_DIRNAME)):
    if not filename.startswith("wsj_"):
        continue
    path = os.path.join(RAW_DIRNAME, filename)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and line != ".START":
                print(line)
                total_count += 1

sys.stderr.write("Total sentences: %d\n" % total_count)
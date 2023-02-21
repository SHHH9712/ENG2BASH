import os
import re
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

input = []
output = []
for file_name in os.listdir(dir_path):
    if ".txt" not in file_name:
        continue
    with open(os.path.join(dir_path, file_name)) as file:
        for line in file.readlines():
            m = re.search("[0-9]*[.].", line)
            # print(line, m)
            # print(line[m.end(0):].split(' - '))
            command, description = line[m.end(0) :].split(" - ")
            input.append(command.strip())
            output.append(description.strip())


pd.DataFrame(input).to_csv(
    os.path.join(dir_path, "input.csv"), index=False, header=False
)
pd.DataFrame(output).to_csv(
    os.path.join(dir_path, "output.csv"), index=False, header=False
)

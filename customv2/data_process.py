import os
import re
import pandas as pd

data = {"output": [], "input":[]}
dataset_path = "/Users/qsha/NL2BASH-simple/customv2"
for file_name in os.listdir(dataset_path):
    if '.txt' not in file_name:
        continue
    with open(os.path.join(dataset_path, file_name)) as file:
        for line in file.readlines():
            m = re.search("[0-9]*[.].", line)
            # print(line, m)
            # print(line[m.end(0):].split(' - '))
            command, description = line[m.end(0):].split(' - ')
            data["output"].append(command.strip())
            data["input"].append(description.strip())


pd.DataFrame(data).to_csv(os.path.join(dataset_path, "customv2.csv"))
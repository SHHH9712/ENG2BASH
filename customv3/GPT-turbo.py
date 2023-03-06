import openai
import yaml
import pandas as pd
with open(r"C:\Users\sha\key") as f:
    keys = yaml.safe_load(f)
openai.api_key = keys["openai"]





def generate_command(command):
    content_1 = "Can you give me 100 real example of linux commands = \""
    content_2 = "\" with different mock-up file/directory name, and it's description in this format: {command} -> {description}. Each command should only include less than 5 words, with options separated. Each descriptions should only talks about its function, options and directory."
    print(command)
    content = content_1 + command + content_2
    print(content)
    result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant that can generate linux command example and its description."},
            {"role": "user", "content": content},
        ]
    )


    # return result["choices"][0]["message"]["content"]
    
    with open(command + ".txt", "w") as f:
        f.write(result["choices"][0]["message"]["content"].replace("`", ""))



commands = pd.read_csv(r"C:\Users\sha\Desktop\ENG2BASH\finalized_commnads.csv")
for i in commands["commands"]:
    if i[0] != "-":
        generate_command(i.strip())

import os  
new_word = set()
file_path = "result"
for file_name in os.listdir(file_path):
    file = os.path.join(file_path, file_name)
    with open(file, "r",encoding='UTF-8') as f:
        for line in f:
            info = line.split()
            new_word.add(info[0])
print(len(new_word))
print(new_word)
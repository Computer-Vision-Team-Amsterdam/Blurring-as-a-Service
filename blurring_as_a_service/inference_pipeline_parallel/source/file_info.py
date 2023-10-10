# Simple script that reads files and prints the contents

# TODO dit moet init en run functions bevatten


def init():
    print("no idea what to put here")


def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")

    # Loop over each text file and print its name and contents
    for file_path in mini_batch:
        with open(file_path, "r") as file:
            file_contents = file.read()
            print(f"File Name: {file_path}")
            print(f"File Contents:\n{file_contents}\n")

    onzin_list = [1, 2, 3]
    return onzin_list

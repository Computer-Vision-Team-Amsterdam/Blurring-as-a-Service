from os import listdir


def get_and_store_info(data_folder_path: str):
    content = [f for f in listdir(data_folder_path)]

    with open("validation_content.txt", "w") as f:
        f.writelines(content)

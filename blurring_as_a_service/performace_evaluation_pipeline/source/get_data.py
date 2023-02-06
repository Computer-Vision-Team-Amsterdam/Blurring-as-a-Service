from os import listdir


def get_and_store_info(data_folder_path: str, output_file_path: str):
    content = [f for f in listdir(data_folder_path)]

    with open(output_file_path, "w") as f:
        f.writelines(content)

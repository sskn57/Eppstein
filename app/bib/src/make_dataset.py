import os
import settings

if __name__ == "__main__":
    dataset_name = "DATASET002"
    dataset_dir_path = os.path.join(settings.data_dir_path, dataset_name)
    if os.path.exists(dataset_dir_path):
        exit()
    else:
        os.mkdir(dataset_dir_path)
        input_dir_path = os.path.join(dataset_dir_path, settings.input_dir_name)
        os.mkdir(input_dir_path)
        newfile_paths = [
            os.path.join(input_dir_path, settings.graph_G_name),
            os.path.join(input_dir_path, settings.pos_data_name),
            os.path.join(input_dir_path, settings.settings_data_name)
            ]
        for fpath in newfile_paths:
            f = open(fpath, "w")
            f.write("")
            f.close()
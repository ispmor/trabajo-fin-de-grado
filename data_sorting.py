import wfdb
import os
import shutil


def sort_signals(data_path):
    absolute_data_path = os.path.abspath(os.getcwd() + data_path[1:])
    for i in range(1, 6877):
        file_name = "A" + str(i).zfill(4)
        file = wfdb.io.rdsamp(data_path + file_name)

        header = file[1]
        signal_class = header["comments"][2][4:]

        class_dir_path = os.path.abspath(os.getcwd()) + "/data/" + signal_class
        if not os.path.exists(class_dir_path):
            os.mkdir(class_dir_path)

        if not os.path.exists(class_dir_path + "/" + file_name + ".mat"):
            shutil.copy(absolute_data_path + "/" + file_name + ".mat", class_dir_path)
            shutil.copy(absolute_data_path + "/" + file_name + ".hea", class_dir_path)

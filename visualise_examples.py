import wfdb
import os
import matplotlib.pyplot as plt


def plot_each_example():
    plt.figure(1)
    i = 0
    j = 0
    data_dir = os.getcwd()+"/data/"
    for (dirpath, dirs, files) in os.walk(data_dir):
        for dir in dirs:
            for (dirpath_interior, dirs_interior, files_interior) in os.walk(data_dir + dir + "/"):
                file = wfdb.io.rdsamp(data_dir + dir + "/" + files_interior[0][:-4])
                ax = plt.subplot2grid((6,6), (i,j))
                ax.plot(file[0][:,3])
                ax.set_title(dir)
                j += 1
                if j > 5:
                    j = 0
                    i += 1
                break

    plt.show()



from mnist_sort import Sorter
from gui import *

if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI(root)
    mnist_network = Sorter()
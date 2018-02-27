import os
import uuid
from tkinter import *

from PIL import Image, ImageTk

from config import corp_size, offset, unit, p

root = "./split_sample_labeled"


class Checker(object):
    def __init__(self):

        self.images = list()
        self.image_index = 0

        for fn in os.listdir(root):
            file = os.path.join(root, fn)

            if os.path.isfile(file):
                self.images.append(file)

        self.master = Tk()

        self.w = Canvas(self.master, width=corp_size, height=corp_size)

        self.cut = self.w.create_text(20, 10, fill="red", font=("Consolas", 12), text="Cut")
        self.cloud = self.w.create_text(70, 10, fill="red", font=("Consolas", 12), text="Cloud")
        self.shadow = self.w.create_text(140, 10, fill="red", font=("Consolas", 12), text="Shadow")

        self.is_cut = False
        self.is_cloud = False
        self.is_shadow = False


        self.master.bind("<Key>", self.key)
        self.w.pack()

        self.w.create_line(0, offset, corp_size, offset, fill="red", dash=(8, 8), width=4)
        self.w.create_line(0, offset + unit, corp_size, offset + unit, fill="red", dash=(8, 8), width=4)

        self.w.create_line(offset, 0, offset, corp_size, fill="red", dash=(8, 8), width=4)
        self.w.create_line(offset + unit, 0, offset + unit, corp_size, fill="red", dash=(8, 8), width=4)

        self.photo_image = None
        self.image_on_canvas = None

        self.load_next()

    def load_previous(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load()

    def load_next(self):
        if self.image_index < (len(self.images) - 1):
            self.image_index += 1
            self.load()

    def load(self):

        image_name = self.images[self.image_index]

        self.photo_image = ImageTk.PhotoImage(Image.open(image_name))
        self.image_on_canvas = self.w.create_image((0, 0), anchor=NW, image=self.photo_image)
        self.w.tag_lower(self.image_on_canvas)

        classes = [bool(int(i)) for i in p.search(image_name).group().split("-")]

        if classes[0]:
            self.w.itemconfig(self.cut, fill="green")
        else:
            self.w.itemconfig(self.cut, fill="red")

        if classes[1]:
            self.w.itemconfig(self.cloud, fill="green")
        else:
            self.w.itemconfig(self.cloud, fill="red")

        if classes[2]:
            self.w.itemconfig(self.shadow, fill="green")
        else:
            self.w.itemconfig(self.shadow, fill="red")

    def key(self, event):
        print("pressed", repr(event.char))
        if event.char.__eq__("a"):
            self.load_previous()

        if event.char.__eq__("d"):
            self.load_next()

    def start(self):
        mainloop()


s = Checker()
s.start()

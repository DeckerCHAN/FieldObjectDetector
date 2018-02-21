import os
import uuid
from tkinter import *

from PIL import Image, ImageTk

from config import corp_size, offset, unit

root = "./split_sample"


class Sampler(object):
    def __init__(self):

        self.images = list()

        for fn in os.listdir(root):
            file = os.path.join(root, fn)

            if os.path.isfile(file):
                self.images.append(file)
        self.images = sorted(self.images)

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

    def refresh(self):
        if self.is_cut:
            self.w.itemconfig(self.cut, fill="green")
        else:
            self.w.itemconfig(self.cut, fill="red")

        if self.is_cloud:
            self.w.itemconfig(self.cloud, fill="green")
        else:
            self.w.itemconfig(self.cloud, fill="red")

        if self.is_shadow:
            self.w.itemconfig(self.shadow, fill="green")
        else:
            self.w.itemconfig(self.shadow, fill="red")

    def key(self, event):
        print("pressed", repr(event.char))
        if event.char.__eq__("a"):
            self.is_cut = not self.is_cut

        if event.char.__eq__("s"):
            self.is_cloud = not self.is_cloud

        if event.char.__eq__("d"):
            self.is_shadow = not self.is_shadow

        if event.char.__eq__('\r'):
            if self.image_on_canvas:
                image_name = self.images[-1]
                self.images.pop()
                os.rename(image_name,
                          str.format('./split_sample_labeled/{0}[{1}-{2}-{3}].jpg', str(uuid.uuid4()), int(self.is_cut),
                                     int(self.is_cloud), int(self.is_shadow)))
                self.is_cut = False
                self.is_cloud = False
                self.is_shadow = False
            if len(self.images) >= 1:

                self.master.title(self.images[-1])

                self.photo_image = ImageTk.PhotoImage(Image.open(self.images[-1]))
                self.image_on_canvas = self.w.create_image((0, 0), anchor=NW, image=self.photo_image)
                self.w.tag_lower(self.image_on_canvas)
            else:
                self.master.quit()

        self.refresh()

    def start(self):
        mainloop()


s = Sampler()
s.start()

import tkinter as tk  # 导入tkinter模块
import tkinter.font as tkFont
import os, shutil
from PIL import Image, ImageTk

window = tk.Tk()  # 主窗口
window.title('Image tag')  # 窗口标题
window.geometry('450x270')  # 窗口尺寸

l = tk.Label(window)
l.place(x=25, y=20, width=400, height=160)

ft = tkFont.Font(family='Fixdsys', size=50, weight=tkFont.BOLD)
t = tk.Entry(window, font=ft)  # 创建文本框，用户可输入内容
t.place(x=145, y=200, height=60, width=160)

rootdir = "image_to_tag"
outdir = "taged_image"

list = os.listdir(rootdir)

count = 0


def mymovefile(srcfile, dstfile):
    fpath, fname = os.path.split(dstfile)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    shutil.move(srcfile, dstfile)


def get_next_image():
    global count
    if list[count] != ".DS_Store":
        path = os.path.join(rootdir, list[count])
        img = Image.open(path)
        img = img.resize((400, 160))
        img = ImageTk.PhotoImage(img)
    count += 1
    return img


img = get_next_image()
l.config(image=img)
l.image = img


def image_save():
    mymovefile(os.path.join(rootdir, list[count - 1]), os.path.join(outdir, str(t.get()) + ".png"))

    global img

    img = get_next_image()
    l.config(image=img)
    l.image = img

    t.delete(0, len(str(t.get())))
    t.icursor(len(str(t.get())))


def printkey(event):
    if event.keycode == 2359309:
        if len(str(t.get())) == 4:
            image_save()
    else:

        text = str(t.get())
        text = text.upper()
        t.delete(0, len(text))
        t.insert(0, text)
        t.icursor(len(text))
        if not str(event.char).isalpha():
            t.delete(len(text) - 1, len(text))
            t.icursor(len(text) - 1)
            return
        if len(text) > 4:
            t.delete(4, len(text))
            t.icursor(4)
            return


window.bind('<Key>', printkey)

window.mainloop()  # 循环消息，让窗口活起来

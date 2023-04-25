from PIL import ImageTk, Image, ImageDraw
import PIL
import customtkinter
import numpy as np

customtkinter.set_appearance_mode("system")

WHITE = (255, 255, 255)


class GUI:
    def __init__(self, recognize):
        self.recognize_digit = recognize

        self.width = 600
        self.height = 548

        self.root = customtkinter.CTk()
        self.root.geometry(f'{self.width}x{self.height}')
        self.root.resizable(False, False)

        self.num = []
        self.per = []
        self.prediction = [0 for i in range(10)]

        #PIL image
        self.image = PIL.Image.new("RGB", (360, 360), WHITE)
        self.draw = ImageDraw.Draw(self.image)

        # frame
        self.frame = customtkinter.CTkFrame(master = self.root)
        self.frame.pack(pady = 20, padx = 60, fill = "both", expand = True)

        self.label = customtkinter.CTkLabel(master = self.frame, text = "Draw a digit")
        self.label.pack(pady = 12, padx = 10)

        # cv_frame
        self.cv_frame = customtkinter.CTkFrame(master = self.frame)
        self.cv_frame.pack()

        self.cv = customtkinter.CTkCanvas(master = self.cv_frame, width = 360, height = 360, bg = "white")
        self.cv.grid(row = 0, column = 0)
        self.cv.bind("<B1-Motion>", self.paint)
        self.px, self.py = None, None

        self.cv.bind("<ButtonRelease-1>", self.on_button_release)

        self.expand_button = customtkinter.CTkButton(master = self.cv_frame, text = ">", width = 28, command = self.expand_window)
        self.expand_button.grid(row = 0, column = 2, pady = 12, padx = 12)

        # num_frame
        self.num_frame = customtkinter.CTkFrame(master = self.cv_frame)
        self.num_frame.pack_forget()

        for i in range(10):
            self.num.append(customtkinter.CTkLabel(master = self.num_frame, text = i))
            self.num[i].grid(padx = 10, row = i, column = 0)

            self.per.append(customtkinter.CTkLabel(master = self.num_frame, text = f'{self.prediction[i] * 100:.0f}%'))
            self.per[i].grid(padx = 10, row = i, column = 1)

        # button_frame
        self.button_frame = customtkinter.CTkFrame(master = self.frame, width = 364)
        self.button_frame.pack(pady= 20, padx = 32, side = customtkinter.LEFT)

        self.recognize_button = customtkinter.CTkButton(master = self.button_frame, text = "Recognize", command = self.recognize)
        self.recognize_button.grid(row = 0, column = 0, pady = 12, padx = 12)

        self.clear_button = customtkinter.CTkButton(master = self.button_frame, text = "Clear", command = self.clear)
        self.clear_button.grid(row = 0, column = 1, pady = 12, padx = 12)

        self.guess_label = customtkinter.CTkLabel(master = self.button_frame, text = f'Guess: {np.argmax(self.prediction)}')
        self.guess_label.pack_forget()

        self.root.mainloop()

    def recognize(self):
        self.prediction = self.recognize_digit(self.image)
        self.update()

        self.open()
        self.show_guess()

    def on_button_release(self, event):
        self.px = None
        self.py = None

    def clear(self):
        self.cv.delete("all")
        self.draw.rectangle((0, 0, self.width, self.height), fill=WHITE)
        self.close()
        self.hide_guess()

    def paint(self, event):
        if self.px is None or self.py is None: 
            self.px, self.py = event.x, event.y
        self.cv.create_line(event.x, event.y, self.px, self.py, fill='black', width=15)
        self.draw.line((event.x, event.y, self.px, self.py), fill='black')
        self.px, self.py = event.x, event.y

    def expand_window(self):
        if self.width < 672:
            return self.open()
        self.close()

    def open(self):
        self.width = 672
        self.expand_button.configure(text = "<")
        self.root.geometry(f'{self.width}x{self.height}')
        self.num_frame.grid(row = 0, column = 1)

    def close(self):
        self.width = 600
        self.expand_button.configure(text = ">")
        self.root.geometry(f'{self.width}x{self.height}')
        self.num_frame.grid_forget()

    def show_guess(self):
        self.guess_label.grid(row = 0, column = 2, pady = 12, padx = 19)

    def hide_guess(self):
        self.guess_label.grid_forget()

    def update(self):
        for i in range(10):
            self.per[i].configure(text = f'{self.prediction[i] * 100:.0f}%')
        self.guess_label.configure(text = f'Guess: {np.argmax(self.prediction)}')


if __name__ == '__main__':
    gui = GUI(lambda x: [.1, 1, 1, 1, 1, 1,1 ,1 ,1,1])
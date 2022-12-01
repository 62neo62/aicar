import tkinter as tk
#from startstopcar import *
import back_wheels
from back_wheels.py import stop
import main
from main import main
from tkinter import *

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)        
        self.master = master

        # widget can take all window
        self.pack(fill=BOTH, expand=1)

        # create button, link it to clickExitButton()
        exitButton = Button(self, text="Exit", command=self.clickExitButton, bg="red",fg="white")

        # place button at (0,0)
        exitButton.place(x=0, y=0)

        startcarbutton = tk.Button(text="Fire up the engine!",command=startbutton, width=25,height=10,bg="green",fg="white")
        # place button at (0,0)
        startcarbutton.place(x=65, y=50)

        stopcarbutton = tk.Button(text="Stop the car!",command =self.stopbutton, width=25,height=10,bg="red",fg="white")
        # place button at (0,0)
        stopcarbutton.place(x=65, y=200)

    def clickExitButton(self):
        exit()
    def stopbutton():
        stop(self)
    
    def startbutton():
        main()
        
root = Tk()
app = Window(root)
root.wm_title("AIcar")
root.geometry("300x400")
root.mainloop()

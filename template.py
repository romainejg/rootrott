#template from BoxingApp. has a button to run script, a back button with functionality, and proper design

import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
from tkinter import PhotoImage
from PIL import Image, ImageTk
import os

class BoxingApp:
    def __init__(self, root, back_callback):
        self.root = root
        self.back_callback = back_callback
        self.create_initial_interface()

    def clear_root_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_initial_interface(self):
        self.clear_root_widgets()

        self.header_font = Font(family="Helvetica", size=28, weight="bold")
        self.subtext_font = Font(family="Helvetica", size=16, slant="italic")
        self.button_font = Font(family="Helvetica", size=16, weight="bold")

        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill="both", expand=True)

        header_label = ttk.Label(frame, text="Boxing Script", font=self.header_font)
        header_label.pack(pady=10)

        subtext_label = ttk.Label(frame, text="Select an option below to start", font=self.subtext_font)
        subtext_label.pack(pady=10)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(script_dir, "assets", "logo.png")

        try:
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((300, 300), Image.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(frame, image=logo_photo)
            logo_label.image = logo_photo
            logo_label.pack(pady=20)
        except Exception as e:
            print(f"Error loading logo image: {e}")

        style = ttk.Style()
        style.configure('TButton', font=self.button_font, padding=10)

        run_button = ttk.Button(frame, text="Run script", command=self.placeholder_function, style='TButton')
        run_button.pack(pady=10, fill="x")

        back_button = ttk.Button(frame, text="Back", command=self.back_to_main, style='TButton')
        back_button.pack(pady=10, fill="x")

    def placeholder_function(self):
        print("Button pressed (no functionality yet)")

    def back_to_main(self):
        self.clear_root_widgets()
        self.back_callback()

def main_interface(root, back_callback):
    BoxingApp(root, back_callback)

if __name__ == "__main__":
    root = tk.Tk()
    app = BoxingApp(root, lambda: print("Back button pressed"))
    root.mainloop()

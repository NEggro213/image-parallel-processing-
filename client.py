import requests
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing")

        self.label = tk.Label(master, text="Select an image and operation:", bg="#f0f0f0")
        self.label.pack(pady=10)

        self.image_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.image_button.pack(pady=5)

        self.operation_label = tk.Label(master, text="Select Operation:", bg="#f0f0f0")
        self.operation_label.pack(pady=5)

        self.operations = ["edge_detection", "color_inversion", "superpixel_segmentation", "gaussian_blur",
                           "otsu_threshold"]  # Otsu's threshold added
        self.operation_var = tk.StringVar(master)
        self.operation_var.set(self.operations[0])  # Default operation
        self.operation_option = tk.OptionMenu(master, self.operation_var, *self.operations)
        self.operation_option.pack(pady=5)

        self.process_button = tk.Button(master, text="Process", command=self.process)
        self.process_button.pack(pady=5)

        self.save_button = tk.Button(master, text="Save Image", command=self.save_image, state="disabled")
        self.save_button.pack(pady=5)

        self.processed_image_label = tk.Label(master, bg="#f0f0f0")
        self.processed_image_label.pack(padx=10, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            messagebox.showinfo("Image Selected", "Image selected successfully!")
            self.save_button.config(state="disabled")  # Disable save button when a new image is selected
            self.processed_image_label.config(image=None)  # Clear previous processed image

    def process(self):
        if hasattr(self, 'image_path'):
            operation = self.operation_var.get()
            result = self.send_request(self.image_path, operation)
            if result is not None:
                self.display_processed_image(result)
                self.save_button.config(state="normal")  # Enable save button after processing
        else:
            messagebox.showerror("Error", "Please select an image first!")

    def send_request(self, image_path, operation):
        with open(image_path, 'rb') as file:
            image_data = file.read()

        url = 'http://102.37.18.28:5000/process_image'
        files = {'image': ('image.jpg', image_data)}
        data = {'operation': operation}
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result_data = response.content
            result_array = np.frombuffer(result_data, dtype=np.uint8)
            result_image = cv2.imdecode(result_array, cv2.IMREAD_COLOR)
            return result_image
        else:
            messagebox.showerror("Error", "Failed to process image.")
            return None

    def display_processed_image(self, result):
        processed_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        processed_img = ImageTk.PhotoImage(processed_img)
        self.processed_image_label.config(image=processed_img)
        self.processed_image_label.image = processed_img  # Keep a reference to avoid garbage collection

    def save_image(self):
        if hasattr(self, 'image_path'):
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if file_path:
                try:
                    processed_image = self.processed_image_label.image
                    processed_image = processed_image._PhotoImage__photo.subsample(1, 1)  # To save it properly
                    processed_image.write(file_path, format="png")  # Change 'jpeg' to 'jpg'
                    messagebox.showinfo("Image Saved", "Image saved successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {e}")
        else:
            messagebox.showerror("Error", "Please process an image first!")


def main():
    root = tk.Tk()
    GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
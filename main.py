from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Frame, Canvas
from tkinter.font import Font

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Function to handle image selection
def select_image():
    global image_path
    image_path = filedialog.askopenfilename(
        initialdir="/",
        title="Select Image",
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*"))
    )
    update_image_label(image_path)

# Function to update the image label in the GUI
def update_image_label(path):
    image_label.config(text="Selected Image: {}".format(path))

# Function to perform prediction on the selected image
def predict_image():
    if image_path:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Perform prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Update the result label in the GUI
        result_label.config(text="Class: {} | Confidence Score: {:.2%}".format(class_name[2:], confidence_score))

# Create the main window
root = Tk()
root.title("Image Classifier")

# Styling
font = Font(family="Helvetica", size=12)
bg_color = "#f0f0f0"
frame_bg_color = "#d9d9d9"
button_color = "#4caf50"
button_hover_color = "#45a049"

# Main frame
main_frame = Frame(root, bg=bg_color)
main_frame.pack(expand=True, fill="both")

# Create and place GUI components
select_button = Button(
    main_frame,
    text="Select Image",
    command=select_image,
    font=font,
    bg=button_color,
    fg="white",
    activebackground=button_hover_color,
    padx=10,
    pady=5
)
select_button.pack(pady=10)

image_label = Label(
    main_frame,
    text="Selected Image: None",
    font=font,
    bg=frame_bg_color
)
image_label.pack()

predict_button = Button(
    main_frame,
    text="Predict Image",
    command=predict_image,
    font=font,
    bg=button_color,
    fg="white",
    activebackground=button_hover_color,
    padx=10,
    pady=5
)
predict_button.pack(pady=10)

result_label = Label(
    main_frame,
    text="Class: None | Confidence Score: None",
    font=font,
    bg=frame_bg_color
)
result_label.pack()

# Initialize image_path variable
image_path = None

# Run the Tkinter event loop
root.mainloop()
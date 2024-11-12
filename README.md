ImageProcessorApp
ImageProcessorApp is a tkinter-based image processing application written in Python. It allows users to load and process images from a selected folder, applying various image processing techniques, and save the processed images.
Classes
ImageProcessorApp
The main class to initialize and run the image processing application.
Methods
__init__(self, master) Initializes the main application window and variables.
master: The root window in tkinter.
create_widgets(self) Creates and packs all the widgets (buttons, canvas, frames) in the main window.
load_folder(self) Opens a file dialog to select a folder containing images. Loads the list of supported image files from the selected folder.
load_image(self) Loads the current image based on current_image_index from the image_list. Uses OpenCV to read the image.
display_image(self, image) Displays the provided image on the canvas using tkinter-compatible format.
image: The image to be displayed (numpy array).
show_prev_image(self) Displays the previous image in the image_list.
show_next_image(self) Displays the next image in the image_list.
histogram_equalization(self) Applies histogram equalization to the current image for contrast enhancement.
linear_contrast(self) Applies a linear contrast adjustment to the current image.
order_stat_filter(self) Applies a non-linear filter (e.g., median filter) to the current image.
save_image(self) Opens a save file dialog to save the processed image.
Usage
To run the application, execute the script. It initializes the ImageProcessorApp and opens a tkinter window.
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()

    Dependencies
tkinter: Python's standard GUI library
PIL: Python Imaging Library for image processing (Pillow needs to be installed)
os: Standard Python library for interacting with the operating system
cv2: OpenCV library for computer vision tasks
Installation
First, ensure you have the required libraries installed. You can install Pillow and OpenCV using pip:
pip install pillow opencv-python
Functionality
Folder Selection: Allows users to select a folder containing images.
Image Navigation: Navigate through the images in the selected folder using "Previous" and "Next" buttons.
Image Processing: Apply different processing techniques like histogram equalization, linear contrast adjustment, and a non-linear filter.
Save Processed Image: Save the processed image to a specified location.
GUI Elements
Buttons:
Выбрать папку (Select Folder): Opens the folder selection dialog.
Предыдущее (Previous) and Следующее (Next): Navigate through images.
Гист. эквализация (Histogram Equalization), Лин. контраст (Linear Contrast), Нелин. фильтр (Non-linear Filter): Apply respective image processing techniques.
Сохранить (Save): Save the currently processed image.
Canvas:
Displays the currently loaded or processed image.


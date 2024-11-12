import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Обработка изображений")
        self.master.geometry("1300x900")  # Увеличена ширина для удобного отображения

        # Инициализация переменных
        self.image_folder = ""
        self.image_list = []
        self.current_image_index = 0
        self.original_image = None  # Исходное изображение
        self.base_image = None  # Изображение после яркости и контрастности
        self.processed_image = None  # Изображение после применения фильтров

        # Параметры для линейного контрастирования
        self.alpha = 1.0  # Контраст
        self.beta = 0  # Яркость

        # Параметры для нелинейных фильтров
        self.filter_type = "Медианный"
        self.kernel_size = 3

        # Переменная для выбора метода эквализации гистограммы
        self.equalize_method = tk.StringVar()
        self.equalize_method.set("RGB")  # Значение по умолчанию

        # Список фильтров
        self.filters = []

        # Создание элементов интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Меню выбора папки и навигации
        self.top_frame = tk.Frame(self.master)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.folder_btn = tk.Button(self.top_frame, text="Выбрать папку", command=self.load_folder)
        self.folder_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.prev_btn = tk.Button(self.top_frame, text="Предыдущее", command=self.show_prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(self.top_frame, text="Следующее", command=self.show_next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # Область отображения изображений
        self.display_frame = tk.Frame(self.master)
        self.display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Исходное изображение
        self.original_canvas = tk.Canvas(self.display_frame, width=500, height=500, bg='gray')
        self.original_canvas.pack(side=tk.LEFT, padx=10, pady=10)

        # Обработанное изображение
        self.processed_canvas = tk.Canvas(self.display_frame, width=500, height=500, bg='gray')
        self.processed_canvas.pack(side=tk.RIGHT, padx=10, pady=10)

        # Гистограммы
        self.hist_frame = tk.Frame(self.master)
        self.hist_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.original_hist_canvas = FigureCanvasTkAgg(Figure(figsize=(5, 2)), master=self.hist_frame)
        self.original_hist_canvas.get_tk_widget().pack(side=tk.LEFT, padx=10)

        self.processed_hist_canvas = FigureCanvasTkAgg(Figure(figsize=(5, 2)), master=self.hist_frame)
        self.processed_hist_canvas.get_tk_widget().pack(side=tk.RIGHT, padx=10)

        # Панель инструментов
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Линейное контрастирование
        contrast_frame = tk.LabelFrame(self.controls_frame, text="Линейное контрастирование")
        contrast_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        tk.Label(contrast_frame, text="Контраст (alpha)").grid(row=0, column=0, padx=5, pady=5)
        self.alpha_slider = tk.Scale(contrast_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
                                     command=self.update_alpha)
        self.alpha_slider.set(1.0)
        self.alpha_slider.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(contrast_frame, text="Яркость (beta)").grid(row=1, column=0, padx=5, pady=5)
        self.beta_slider = tk.Scale(contrast_frame, from_=-100, to=100, resolution=1, orient=tk.HORIZONTAL,
                                    command=self.update_beta)
        self.beta_slider.set(0)
        self.beta_slider.grid(row=1, column=1, padx=5, pady=5)

        self.apply_contrast_btn = tk.Button(contrast_frame, text="Применить контрастирование",
                                            command=self.apply_linear_contrast)
        self.apply_contrast_btn.grid(row=2, column=0, columnspan=2, pady=5)

        # Выравнивание гистограммы
        hist_eq_frame = tk.LabelFrame(self.controls_frame, text="Выравнивание гистограммы")
        hist_eq_frame.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Радиокнопки для выбора метода эквализации
        self.rgb_radio = tk.Radiobutton(hist_eq_frame, text="RGB пространство", variable=self.equalize_method,
                                        value="RGB")
        self.rgb_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.hsv_radio = tk.Radiobutton(hist_eq_frame, text="HSV пространство", variable=self.equalize_method,
                                        value="HSV")
        self.hsv_radio.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.apply_hist_eq_btn = tk.Button(hist_eq_frame, text="Применить эквализацию гистограммы",
                                           command=self.apply_histogram_equalization)
        self.apply_hist_eq_btn.grid(row=2, column=0, padx=5, pady=5)

        # Нелинейные фильтры
        filter_frame = tk.LabelFrame(self.controls_frame, text="Нелинейные фильтры")
        filter_frame.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        tk.Label(filter_frame, text="Тип фильтра").grid(row=0, column=0, padx=5, pady=5)
        self.filter_combo = ttk.Combobox(filter_frame, values=["Медианный", "Минимальный", "Максимальный"])
        self.filter_combo.current(0)
        self.filter_combo.grid(row=0, column=1, padx=5, pady=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self.update_filter_type)

        tk.Label(filter_frame, text="Размер ядра").grid(row=1, column=0, padx=5, pady=5)
        self.kernel_slider = tk.Scale(filter_frame, from_=3, to=15, resolution=2, orient=tk.HORIZONTAL,
                                      command=self.update_kernel_size)
        self.kernel_slider.set(3)
        self.kernel_slider.grid(row=1, column=1, padx=5, pady=5)

        self.apply_filter_btn = tk.Button(filter_frame, text="Применить фильтр", command=self.apply_non_linear_filter)
        self.apply_filter_btn.grid(row=2, column=0, columnspan=2, pady=5)

        # Кнопки сохранения и сброса
        save_reset_frame = tk.Frame(self.controls_frame)
        save_reset_frame.grid(row=0, column=3, padx=10, pady=10, sticky="w")

        self.save_btn = tk.Button(save_reset_frame, text="Сохранить", command=self.save_image)
        self.save_btn.pack(side=tk.TOP, padx=5, pady=5)

        self.reset_btn = tk.Button(save_reset_frame, text="Сброс", command=self.reset_image)
        self.reset_btn.pack(side=tk.TOP, padx=5, pady=5)

    def load_folder(self):
        self.image_folder = filedialog.askdirectory()
        if self.image_folder:
            # Получение списка файлов изображений
            supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            self.image_list = [f for f in os.listdir(self.image_folder) if f.lower().endswith(supported_formats)]
            self.image_list.sort()
            if self.image_list:
                self.current_image_index = 0
                self.filters = []  # Очистка списка фильтров при загрузке новой папки
                self.reset_contrast_brightness()
                self.load_image()
            else:
                messagebox.showerror("Ошибка", "В папке нет изображений поддерживаемых форматов.")

    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_image_index])
        # Используем OpenCV для загрузки изображения
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {image_path}")
            return
        self.reset_modifications()
        self.base_image = self.original_image.copy()
        self.processed_image = self.original_image.copy()
        self.display_images()
        self.plot_histograms()

    def reset_modifications(self):
        # Сброс параметров яркости и контрастности
        self.alpha = 1.0
        self.beta = 0
        self.alpha_slider.set(self.alpha)
        self.beta_slider.set(self.beta)
        # Очистка списка фильтров
        self.filters = []

    def reset_contrast_brightness(self):
        # Сброс параметров яркости и контрастности без очистки фильтров
        self.alpha = 1.0
        self.beta = 0
        self.alpha_slider.set(self.alpha)
        self.beta_slider.set(self.beta)

    def display_images(self):
        # Отображение исходного изображения
        self.display_image_on_canvas(self.original_image, self.original_canvas)
        # Отображение обработанного изображения
        self.display_image_on_canvas(self.processed_image, self.processed_canvas)

    def display_image_on_canvas(self, image, canvas):
        # Масштабирование изображения до размеров Canvas, сохраняя пропорции
        h, w = image.shape[:2]
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width == 1 or canvas_height == 1:
            # Если Canvas еще не инициализирован, устанавливаем размеры по умолчанию
            canvas_width = 500
            canvas_height = 500
        ratio = min(canvas_width / w, canvas_height / h)
        new_width = int(w * ratio)
        new_height = int(h * ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Преобразование изображения для Tkinter
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        # Очистка Canvas и отображение изображения
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=image_tk)
        canvas.image_tk = image_tk  # Сохранение ссылки

    def plot_histograms(self):
        # Очистка предыдущих гистограмм
        self.original_hist_canvas.figure.clear()
        self.processed_hist_canvas.figure.clear()

        # Построение гистограммы для исходного изображения
        self.plot_histogram(self.original_image, self.original_hist_canvas, method="RGB", original=True)

        # Построение гистограммы для обработанного изображения
        # Определяем текущий метод эквализации для правильного построения гистограммы
        method = self.equalize_method.get()
        self.plot_histogram(self.processed_image, self.processed_hist_canvas, method=method, original=False)

    def plot_histogram(self, image, canvas, method, original=True):
        ax = canvas.figure.add_subplot(111)
        ax.clear()  # Очистка предыдущего графика

        if method == "RGB":
            # Конвертация изображения в формат RGB для гистограммы
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color = ('r', 'g', 'b')
            for i, col in enumerate(color):
                hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
            ax.set_xlim([0, 256])
            ax.set_title("Исходное" if original else "Обработанное")
        elif method == "HSV":
            # Конвертация изображения в HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            # Выравнивание гистограммы только для компоненты яркости
            hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
            ax.plot(hist_v, color='k')  # Черный цвет для компоненты V
            ax.set_xlim([0, 256])
            ax.set_title("Исходное" if original else "Обработанное")
        else:
            ax.set_title("Гистограмма")

        canvas.draw()

    def show_prev_image(self):
        if self.image_list:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
            self.load_image()

    def show_next_image(self):
        if self.image_list:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
            self.load_image()

    def update_alpha(self, val):
        self.alpha = float(val)
        self.apply_modifications()

    def update_beta(self, val):
        self.beta = int(val)
        self.apply_modifications()

    def apply_linear_contrast(self):
        if self.original_image is not None:
            # Обновляем параметры и пересчитываем обработанное изображение
            self.apply_modifications()

    def apply_modifications(self):
        if self.original_image is None:
            return

        # Шаг 1: Применяем яркость и контрастность к оригинальному изображению
        self.base_image = cv2.convertScaleAbs(self.original_image, alpha=self.alpha, beta=self.beta)

        # Шаг 2: Применяем все фильтры из списка
        self.processed_image = self.base_image.copy()
        for filt in self.filters:
            if filt['type'] == "Медианный":
                self.processed_image = cv2.medianBlur(self.processed_image, filt['kernel_size'])
            elif filt['type'] == "Минимальный":
                kernel = np.ones((filt['kernel_size'], filt['kernel_size']), np.uint8)
                self.processed_image = cv2.erode(self.processed_image, kernel)
            elif filt['type'] == "Максимальный":
                kernel = np.ones((filt['kernel_size'], filt['kernel_size']), np.uint8)
                self.processed_image = cv2.dilate(self.processed_image, kernel)
            elif filt['type'] == "RGB":
                self.processed_image = self.equalize_hist_rgb(self.processed_image)
            elif filt['type'] == "HSV":
                self.processed_image = self.equalize_hist_hsv(self.processed_image)
            else:
                messagebox.showerror("Ошибка", f"Неизвестный фильтр: {filt['type']}")

        # Обновляем отображение
        self.display_images()
        self.plot_histograms()

    def apply_histogram_equalization(self):
        if self.original_image is not None:
            method = self.equalize_method.get()
            # Удаляем предыдущие эквализации гистограммы, чтобы избежать накопления
            self.filters = [filt for filt in self.filters if filt['type'] not in ['RGB', 'HSV']]
            if method == "RGB":
                # Добавляем фильтр эквализации в RGB
                self.filters.append({'type': 'RGB'})
            elif method == "HSV":
                # Добавляем фильтр эквализации в HSV
                self.filters.append({'type': 'HSV'})
            else:
                messagebox.showerror("Ошибка", "Неизвестный метод эквализации.")
                return
            self.apply_modifications()

    def equalize_hist_rgb(self, image):
        # Разделение на каналы
        b, g, r = cv2.split(image)
        # Выравнивание гистограммы для каждого канала
        eq_b = cv2.equalizeHist(b)
        eq_g = cv2.equalizeHist(g)
        eq_r = cv2.equalizeHist(r)
        # Объединение каналов обратно
        return cv2.merge((eq_b, eq_g, eq_r))

    def equalize_hist_hsv(self, image):
        # Конвертация в HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Выравнивание гистограммы только для компоненты яркости
        eq_v = cv2.equalizeHist(v)
        # Объединение каналов обратно
        hsv_eq = cv2.merge((h, s, eq_v))
        # Конвертация обратно в BGR
        return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    def update_filter_type(self, event):
        self.filter_type = self.filter_combo.get()

    def update_kernel_size(self, val):
        self.kernel_size = int(val)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1  # Размер ядра должен быть нечетным

    def apply_non_linear_filter(self):
        if self.original_image is not None:
            method = self.filter_type
            # Добавляем фильтр в список
            self.filters.append({'type': method, 'kernel_size': self.kernel_size})
            # Применяем все модификации
            self.apply_modifications()

    def reset_image(self):
        if self.original_image is not None:
            # Сброс яркости и контрастности
            self.reset_contrast_brightness()
            # Очистка списка фильтров
            self.filters = []
            # Применяем сброшенные параметры
            self.apply_modifications()

    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if save_path:
                cv2.imwrite(save_path, self.processed_image)
                messagebox.showinfo("Сохранение", "Изображение успешно сохранено.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()

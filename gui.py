import io
import random
import tkinter as tk
from threading import Thread
from tkinter import ttk

import mnist_sort

neural_network = mnist_sort.Sorter()

class GUI(tk.Frame):
    base_color1 = "#ff7979"
    base_color2 = "#636B2F"
    accent_color1 = "#D4DE95"
    accent_color2 = "#eb4d4b"
    darkgray_color = "#8A8A8A"
    lightgray_color = "#D3D3D3"

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.neural_network = neural_network
        self.thread = None

        horizontal_padding = 20

        root = self.parent
        root.configure(background=self.lightgray_color)
        root.title("MNIST Sort")

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry(f"{screen_width}x{screen_height}")

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        # Create two frames for smaller windows
        frame_left = tk.Frame(root, bg=self.lightgray_color, relief="solid", borderwidth=0)
        frame_right = tk.Frame(root, bg=self.lightgray_color, relief="solid", borderwidth=0)

        frame_left.grid_rowconfigure(0, weight=1)
        frame_left.grid_rowconfigure(1, weight=4)
        frame_left.grid_rowconfigure(2, weight=6)
        frame_left.grid_columnconfigure(0, weight=1)
        frame_right.grid_rowconfigure(0, weight=1)
        frame_right.grid_rowconfigure(1, weight=5)
        frame_right.grid_columnconfigure(0, weight=1)
        frame_left.grid_propagate(False)
        frame_right.grid_propagate(False)

        frame_left.grid(row=0, column=0, sticky="nsew", padx=(horizontal_padding, horizontal_padding // 2), pady=20)
        frame_right.grid(row=0, column=1, sticky="nsew", padx=(horizontal_padding // 2, horizontal_padding), pady=20)

        ## LEFT FRAME
        # Title frame
        self.title_frame = BetterFrame(frame_left, self.accent_color2)
        self.title_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        self.title_frame.canvas.bind("<Configure>", self.on_resize)

        # Results Frame
        results_frame = BetterFrame(frame_left, self.accent_color2)
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))
        self.results = Results(results_frame, self.accent_color2)
        self.results.grid(row=0, column=0, sticky="nsew")


        # Progress Frame
        progress_frame = BetterFrame(frame_left, self.accent_color2)
        self.progress_bar = Progress(progress_frame.canvas, self.accent_color2)
        self.progress_bar.run_button.configure(command=lambda: self.start_gradient_descent())
        progress_frame.canvas.rowconfigure(0, weight=1)
        progress_frame.canvas.columnconfigure(0, weight=1)
        progress_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.progress_bar.grid(row=0, column=0, sticky="nsew")
        progress_frame.grid_propagate(False)

        ## RIGHT FRAME
        # Model Parameters
        parameters_frame = BetterFrame(frame_right, self.accent_color2)
        parameters_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        parameters_frame.canvas.grid_rowconfigure(0, weight=1)
        parameters_frame.canvas.grid_columnconfigure(0, weight=1)
        self.parameters = Parameters(parameters_frame.canvas, self.accent_color2)
        self.parameters.grid(row=0, column=0, sticky="nsew")

        # Visualization Frame
        self.visualization_frame = BetterFrame(frame_right, self.accent_color2)
        self.visualization_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.visualization_frame.canvas.grid_rowconfigure(0, weight=1)
        self.visualization_frame.canvas.grid_rowconfigure(1, weight=9)
        self.visualization_frame.canvas.grid_columnconfigure(0, weight=1)
        nodes_per_layer = [784, 128, 10]  # Nodes in each layer
        self.visualization_label = tk.Canvas(self.visualization_frame.canvas)
        self.visualization = NeuralNetworkVisualizer(self.visualization_frame.canvas, nodes_per_layer=[784, 128, 10], draw_connections=True, relief="sunken")
        self.visualization_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20,10))
        self.visualization.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10,20))

        self.progress_bar.progress_bar.update_progress(self.progress_bar.progress)

        root.mainloop()

    def on_resize(self, event):
        self.parent.update_idletasks()
        self.parent.update()

        # Configure title text
        self.title_frame.canvas.delete("all")
        self.title_frame.on_resize(event)
        self.title_frame.canvas.create_rectangle(20, 20, self.title_frame.canvas.winfo_width() - 20,
                                                 self.title_frame.canvas.winfo_height() - 20, fill=self.base_color1,
                                                 outline="black", width=3)
        self.create_outlined_text(self.title_frame.canvas, self.title_frame.canvas.winfo_width() / 2,
                                  self.title_frame.canvas.winfo_height() / 2, text="MNIST Neural Network",
                                  outline_color="black", text_color="white", outline_width=2,
                                  font=("Arial bold", 60))

        # Visualization text label
        self.visualization_label.configure(height=self.visualization_frame.winfo_height() / 10,
                                           bg=self.base_color1)
        GUI.create_shadow_text(self.visualization_label, self.visualization_frame.canvas.winfo_width() / 2, 50,
                        "Network Visualization", shadow_color="black", text_color="white", font="Arial 40 bold")

    # Called after every epoch
    def update_gui(self):
        self.visualization.update_output_color()
        self.parent.update_idletasks()
        self.parent.update()

    @staticmethod
    def create_outlined_text(canvas, x, y, text, font, outline_color, text_color, outline_width=1):
        for i in range(-outline_width, outline_width + 1):
            for j in range(-outline_width, outline_width + 1):
                if i != 0 or j != 0:
                    canvas.create_text(x + i, y + j, text=text, font=font, fill=outline_color)
        canvas.create_text(x + i, y + j, text=text, font=font, fill=text_color)
        return

    @staticmethod
    def create_shadow_text(canvas, x, y, text, font, shadow_color, text_color):
        canvas.create_text(x + 2, y + 2, text=text, fill=shadow_color, font=font)
        canvas.create_text(x, y, text=text, fill=text_color, font=font)

    def start_gradient_descent(self):
        try:
            int(self.parameters.nodes_box.entry_value.get())
        except ValueError:
            print("Invalid integer value")
            return

        # Create a thread to run the gradient descent function
        if self.thread is None:
            self.thread = Thread(target=neural_network.gui_gradient_descent, args=(self, self.progress_bar,
            int(self.parameters.epoch_box.entry_value.get()),float(self.parameters.learning_rate_box.entry_value.get()),
                                    int(self.parameters.nodes_box.entry_value.get())))
            self.thread.start()

class NeuralNetworkVisualizer(tk.Frame):
    def __init__(self, parent, nodes_per_layer, draw_connections, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.output_circles = []
        self.draw_connections = draw_connections

        # Canvas for drawing
        self.canvas = tk.Canvas(self, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.layers = len(nodes_per_layer)
        self.nodes_per_layer = nodes_per_layer
        self.node_radius = 16  # Smaller radius for widget
        self.margin_x = 50  # Horizontal margin
        self.margin_y = 50  # Vertical margin

        # Bind resize events to redraw dynamically
        self.bind("<Configure>", lambda e: self.create_visualization())

    def create_visualization(self):
        # Clear the canvas
        self.canvas.delete("all")
        self.update_idletasks()
        self.canvas.configure(background=GUI.lightgray_color)

        # Get the current size of the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate spacing
        if self.layers > 1:
            layer_spacing = (canvas_width - 2 * self.margin_x) / (self.layers - 1)
        else:
            layer_spacing = 0

        node_spacing = 36

        # Store node positions for connections
        node_positions = []

        for layer_idx in range(self.layers): # For each layer
            layer_positions = []
            x = self.margin_x + layer_idx * layer_spacing

            if self.nodes_per_layer[layer_idx] > 10:
                layer_nodes = 10
                y_start = canvas_height / 2 - (layer_nodes - 1) * node_spacing / 2
                for node_idx in range(5):
                    y = y_start + node_idx * node_spacing * 1.2 - 8
                    layer_positions.append((x, y))

                layer_positions.append((x, canvas_height - node_spacing * 1.5 - 8))

            else:
                layer_nodes = self.nodes_per_layer[layer_idx]
                y_start = canvas_height / 2 - (layer_nodes - 1) * node_spacing / 2

                for node_idx in range(layer_nodes):
                    y = y_start + node_idx * node_spacing - 8
                    layer_positions.append((x, y))

            self.canvas.create_text(x+2, canvas_height - 18, fill="black", font="Arial 32 bold",
                                    text=f"{self.nodes_per_layer[layer_idx]}")
            self.canvas.create_text(x, canvas_height - 20, fill="white", font="Arial 32 bold",
                                    text=f"{self.nodes_per_layer[layer_idx]}")

            node_positions.append(layer_positions)

        if self.draw_connections:
            for layer_idx in range(self.layers - 1):
                for start_node in node_positions[layer_idx]:
                    for end_node in node_positions[layer_idx + 1]:
                        self.canvas.create_line(start_node[0], start_node[1], end_node[0], end_node[1], fill="black")

        for layer in range(len(node_positions)):
            if self.nodes_per_layer[layer] > 10: radius = self.node_radius * 1.2
            else: radius = self.node_radius

            if layer == len(node_positions) - 1: count = 0
            for node in node_positions[layer]:
                # Draw node as a circle
                if layer == 0:
                    self.draw_circle(node[0], node[1], radius, fill="#134391", outline="black")
                elif layer == len(node_positions) - 1: # Output
                    y_start = canvas_height / 2 - (10 - 1) * node_spacing / 2
                    number = int((node[1] - y_start + 8) / node_spacing)
                    oval_id = self.draw_circle(node[0], node[1], radius, fill="#333333", outline="black")
                    self.output_circles.append(oval_id)
                    self.canvas.create_text(node[0] + 1, node[1] + 1, fill="black", text=f"{number}")
                    self.canvas.create_text(node[0], node[1], fill="white", text=f"{number}")
                else:
                    self.draw_circle(node[0], node[1], radius, fill="gray", outline="black")

            if self.nodes_per_layer[layer] > 10:
                y_center = node_positions[layer][4][1] + (node_positions[layer][5][1] - node_positions[layer][4][1]) / 2
                self.draw_circle(node_positions[layer][4][0], y_center - 25, 6,
                                                            fill="black", outline="black")
                self.draw_circle(node_positions[layer][4][0], y_center, 6, fill="black", outline="black")
                self.draw_circle(node_positions[layer][4][0], y_center + 25, 6, fill="black", outline="black")

    def update_output_color(self):
        i = 0
        value = random.randint(0, neural_network.A2.shape[1])
        while i < 10:
            self.canvas.itemconfigure(self.output_circles[i], fill=str(ProgressBar.interpolate_color(neural_network.A2[i][value], (51,51,51), (221, 24, 24))))
            i += 1

    def draw_circle(self, x, y, r, fill="white", outline="black"):
        """Draw a circle at (x, y) with radius r."""
        return self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline)

    def draw_rect(self, x0, x1, y0, y1, fill="white", outline="black"):
        """Draw a circle at (x, y) with radius r."""
        return self.canvas.create_rectangle(x0,y0,x1,y1, fill=fill, outline=outline)


class BetterFrame(tk.Frame):
    def __init__(self, parent, color, *args, **kwargs):
        super().__init__(parent, **kwargs)

        self.color = color

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(1, weight=0)

        self.configure(bg=parent.cget("bg"))
        self.canvas = tk.Canvas(self, bg=parent.cget("bg"), highlightthickness=0)
        self.canvas.grid(row=0,column=0,sticky="nsew")

        self.canvas.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        self.canvas.delete("all")  # Clear the canvas

        width = event.width
        height = event.height

        # Create ovals
        self.canvas.create_oval(0, 0, 50, 50, fill=self.color, outline="")  # Top left
        self.canvas.create_oval(width - 50, 0, width, 50, fill=self.color, outline="")  # Top right
        self.canvas.create_oval(0, height - 50, 50, height, fill=self.color, outline="")  # Bottom left
        self.canvas.create_oval(width - 50, height - 50, width, height, fill=self.color, outline="")  # Bottom right
        self.canvas.create_rectangle(0, 25, 50, height-25, fill=self.color, outline="")
        self.canvas.create_rectangle(width-50, 25, width, height-25, fill=self.color, outline="")
        self.canvas.create_rectangle(25, 0, width-25, height, fill=self.color, outline="")

class ShadowLabel(tk.Canvas):
    def __init__(self, parent, font, text, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.font = font
        self.text = text
        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        self.delete("all")
        canvas_width = event.width
        canvas_height = event.height

        self.create_text(canvas_width / 2 + 2, canvas_height / 2 + 2, font=self.font,
                                       fill="black", text=self.text)
        self.create_text(canvas_width / 2, canvas_height / 2, font=self.font, fill="white",
                                      text=self.text)

class Results(BetterFrame):
    def __init__(self, parent, color, **kwargs):
        super().__init__(parent, color, **kwargs)
        self.results_arr = []

        self.canvas.rowconfigure(0, weight=2)
        self.canvas.rowconfigure(1, weight=4)
        self.canvas.columnconfigure(0, weight=1)
        self.canvas.columnconfigure(1, weight=1)
        self.configure(bg="black")

        results_frame = tk.Frame(self, bg="black")
        results_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        results_frame.grid_propagate(False)
        results_frame.rowconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=6)
        results_frame.columnconfigure(0, weight=1)

        self.results_labels = tk.Canvas(results_frame, bg=GUI.lightgray_color, height=60, highlightthickness=2)
        self.results_canvas = tk.Canvas(results_frame, bg=GUI.lightgray_color, highlightthickness=4)
        self.results_labels.grid(row=0, column=0, sticky="nsew", columnspan=1)
        self.results_canvas.grid(row=1, column=0, sticky="nsew", columnspan=1)
        self.results_canvas.grid_propagate(False)

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.results_canvas.configure(yscrollcommand=scrollbar.set)

        scrollable_frame = ttk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        scrollable_frame.bind("<Configure>", self.update_scroll_region)

        # Bind resize events to redraw dynamically
        self.bind("<Configure>", lambda e: self.draw_entries())

    def update_scroll_region(self, event):
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def draw_entries(self):
        self.update_idletasks()
        self.results_labels.delete("all")
        self.results_canvas.delete("all")

        canvas_width = self.results_canvas.winfo_width()
        label_height = self.results_labels.winfo_height()
        value_width = canvas_width / 5

        # Outlines
        self.results_labels.create_rectangle(0, 0, canvas_width, 4, fill="black", outline="")
        self.results_labels.create_rectangle(0, label_height - 2, canvas_width, label_height, fill="black", outline="")
        self.results_labels.create_rectangle(0, 4, 4, label_height - 2, fill="black", outline="")
        self.results_labels.create_rectangle(canvas_width - 4, 4, canvas_width, label_height - 2, fill="black",
                                             outline="")

        # Labels
        GUI.create_shadow_text(self.results_labels, value_width / 2, label_height / 2, text="Run Number",
                               text_color="white", shadow_color="black", font="Arial 20 bold")
        GUI.create_shadow_text(self.results_labels, value_width / 2 + value_width, label_height / 2, text="Parameters",
                               text_color="white", shadow_color="black", font="Arial 20 bold")
        GUI.create_shadow_text(self.results_labels, value_width / 2 + value_width * 2, label_height / 2, text="Loss",
                               text_color="white", shadow_color="black", font="Arial 20 bold")
        GUI.create_shadow_text(self.results_labels, value_width / 2 + value_width * 3, label_height / 2,
                               text="Train\nAccuracy", text_color="white", shadow_color="black", font="Arial 20 bold")
        GUI.create_shadow_text(self.results_labels, value_width / 2 + value_width * 4, label_height / 2,
                               text="Validation\nAccuracy", text_color="white", shadow_color="black",
                               font="Arial 20 bold")

        # For each past run, create a row containing information
        for run_num in range(len(self.results_arr)):
            for value in range(5):
                self.results_canvas.create_rectangle(value_width * value, run_num * 80, value_width * (value + 1),
                                            (run_num + 1) * 80, fill=GUI.lightgray_color, outline="black", width=2)
                self.results_canvas.create_text(value_width * value + value_width / 2, run_num * 80 + 40,
                                anchor="center", text=self.results_arr[run_num][value], fill="black", font="Arial 12")

        self.update_scroll_region(None)

class Progress(BetterFrame):
    def __init__(self, parent, color, **kwargs):
        super().__init__(parent, color, **kwargs)
        self.progress = 0 # 0 - 100

        self.canvas.rowconfigure(0, weight=1)
        self.canvas.rowconfigure(1, weight=8)
        self.canvas.columnconfigure(0, weight=1)
        self.canvas.columnconfigure(1, weight=7)
        self.canvas.grid_propagate(False)

        # Progress Bar
        progress_bar_frame = tk.Frame(self.canvas, bg="black")
        self.progress_bar = ProgressBar(progress_bar_frame, (204, 41, 0), (0, 153, 0))
        progress_bar_frame.rowconfigure(0, weight=1)
        progress_bar_frame.columnconfigure(0, weight=1)

        # Metrics
        self.metrics_frame = tk.Frame(self.canvas, bg=GUI.accent_color2)
        self.metrics_frame.columnconfigure(0, weight=1)
        self.metrics_frame.columnconfigure(1, weight=1)
        self.metrics_frame.columnconfigure(2, weight=1)
        self.metrics_frame.columnconfigure(3, weight=1)
        self.metrics_frame.rowconfigure(0, weight=1)
        self.metrics_frame.grid_propagate(False)

        # StringVar so labels dynamically update
        self.epoch_var = tk.StringVar()
        self.loss_var = tk.StringVar()
        self.train_var = tk.StringVar()
        self.val_var = tk.StringVar()
        self.set_metrics(0, 100, 50, 50)

        epoch = tk.Label(self.metrics_frame, bg="#D3D3D3", textvariable=self.epoch_var, fg="black",
                         font="Arial 16 bold", borderwidth=3, relief="solid")
        loss = tk.Label(self.metrics_frame, bg="#D3D3D3", textvariable=self.loss_var, fg="black", font="Arial 16 bold",
                        borderwidth=3, relief="solid")
        train = tk.Label(self.metrics_frame, bg="#D3D3D3", textvariable=self.train_var,
                         fg="black", font="Arial 16 bold", borderwidth=3, relief="solid")
        val = tk.Label(self.metrics_frame, bg="#D3D3D3", textvariable=self.val_var,
                       fg="black", font="Arial 16 bold", borderwidth=3, relief="solid")

        epoch.grid(row=0, column=0, sticky="nsew", padx=(0,2))
        loss.grid(row=0, column=1, sticky="nsew", padx=2)
        train.grid(row=0, column=2, sticky="nsew", padx=2)
        val.grid(row=0, column=3, sticky="nsew", padx=(2,0))

        # Run button
        self.run_button = tk.Button(self.canvas, text="Run", font=("Arial bold", 36))
        self.run_button.grid_propagate(False)

        progress_bar_frame.grid_propagate(False)
        progress_bar_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(5, 20), columnspan=2)
        self.run_button.grid(row=0, column=0, sticky="nsew", padx=(20,10), pady=(20, 5))
        self.metrics_frame.grid(row=0, column=1, sticky="nsew", padx=(10,20), pady=(20, 5))

        self.progress_bar.grid(row=0, column=0, sticky="nsew")

        self.canvas.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        super().on_resize(event)
        self.progress_bar.update_progress(self.progress)

    def set_metrics(self, epoch, loss, train, validation):
        self.epoch_var.set("Epoch:\n" + str(epoch))
        self.loss_var.set("Loss:\n" + str(round(loss,1)))
        self.train_var.set("Train Accuracy:\n" + str(round(train,1)) + "%")
        self.val_var.set("Validation Accuracy:\n" + str(round(validation,1)) + "%")

class ProgressBar(tk.Frame):
    def __init__(self, parent, color1, color2, **kwargs):
        super().__init__(parent,**kwargs)

        self.color1 = color1
        self.color2 = color2
        self.curr_color = "#000000"
        self.progress = 0 # 0 to 100
        self.rect_id = None
        self.text_id = None
        self.text_id2 = None
        self.canvas = tk.Canvas(self, bg=GUI.lightgray_color)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky="nsew", ipadx=20, ipady=20)

        self.canvas.bind("<Configure>", lambda e: self.update_progress(self.progress))

    def update_progress(self, progress):
        self.progress = progress
        self.curr_color = self.interpolate_color(progress / 100, self.color1, self.color2)

        self.update_idletasks()

        # Get the current size of the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        new_rect_id = self.canvas.create_rectangle(0, 0, canvas_width * self.progress / 100,
                                                   canvas_height,fill=self.curr_color)
        new_text_id = self.canvas.create_text(42, canvas_height / 2 + 2, text=str(self.progress) + "%",
                                              font=("Arial bold", 30), fill="black")
        new_text_id2 = self.canvas.create_text(40, canvas_height / 2, text=str(self.progress) + "%",
                                               font=("Arial bold", 30), fill="white")

        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        if self.text_id is not None:
            self.canvas.delete(self.text_id)
            self.canvas.delete(self.text_id2)

        self.rect_id = new_rect_id
        self.text_id = new_text_id
        self.text_id2 = new_text_id2

    @staticmethod
    def interpolate_color(value, color1, color2):
        hex_color = io.StringIO()
        hex_color.write("#")

        for i in range(3):
            diff = (color2[i] - color1[i]) * value # Find difference and multiply by progress
            if int(color1[i] + diff) < 16: hex_color.write("0")
            hex_color.write('%x' % (int(color1[i] + diff)))

        return hex_color.getvalue()

class Parameters(BetterFrame):
    def __init__(self, parent, color, **kwargs):
        super().__init__(parent, color, **kwargs)

        self._parent = parent
        self.learning_rate = tk.StringVar(value="0.005")
        self.hidden_layers = tk.StringVar(value="64")
        self.epochs = tk.StringVar(value="5")  # Ensure proper initialization

        self.canvas.grid_rowconfigure(0, weight=1)
        self.canvas.grid_columnconfigure(0, weight=1)
        self.canvas.grid_columnconfigure(1, weight=1)
        self.canvas.grid_columnconfigure(2, weight=1)
        self.canvas.grid_propagate(False)

        # Each parameter is an instance of ParameterBox
        self.epoch_box = ParameterBox(self.canvas, self.epochs, "Epoch")
        self.learning_rate_box = ParameterBox(self.canvas, self.learning_rate, "Learning Rate")
        self.nodes_box = ParameterBox(self.canvas, self.hidden_layers, "Hidden Layer Nodes")

        self.epoch_box.grid(row=0, column=0, sticky="nsew", padx=(20,10), pady=20)
        self.learning_rate_box.grid(row=0, column=1, sticky="nsew", padx=10, pady=20)
        self.nodes_box.grid(row=0, column=2, sticky="nsew", padx=(10,20), pady=20)

class ParameterBox(tk.Frame):
    def __init__(self, parent, var, name, **kwargs):
        super().__init__(parent, **kwargs)
        self.entry_value = var # Use StringVar

        self.configure(bg=GUI.accent_color2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid(row=0, column=0, padx=10, pady=10, ipadx=10, ipady=10)
        self.grid_propagate(False)

        self.label = tk.Label(self, text=name, fg="black", bg=GUI.lightgray_color, font=("Arial bold", 24),
                              borderwidth=3, relief="solid")
        self.entry = tk.Entry(self, bg=GUI.lightgray_color, textvariable=self.entry_value, font=("Arial bold", 18),
                              fg="black")
        self.button = tk.Button(self, activebackground=GUI.lightgray_color,
                                fg="black", text="Enter", font=("Arial bold", 20), command=self.read_text)

        self.label.grid(row=0, column=0, sticky="nsew", columnspan=2, pady=(0,4))
        self.label.grid_propagate(False)
        self.entry.grid(row=1, column=0, sticky="nsew", rowspan=1)
        self.entry.grid_propagate(False)
        self.button.grid(row=1, column=1, sticky="nsew", rowspan=1)
        self.button.grid_propagate(False)


    def read_text(self):
        print(f"{self.label.cget("text")} Value: {self.entry_value.get()}")
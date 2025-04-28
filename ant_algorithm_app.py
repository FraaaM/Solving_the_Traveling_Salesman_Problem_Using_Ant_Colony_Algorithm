import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
from tkinter.simpledialog import askinteger
import time  
import random

class TSPApp:
    NODE_SIZE = 12
    MIN_SPACING = 50
    SELECTION_RADIUS = 17
    DEFAULT_PARAMS = {'alpha': 1.0, 'beta': 2.0, 'rho': 0.1, 'q': 100}

    def __init__(self, root):
        self.root = root
        self.root.title("Решение задачи маршрута")
        self.root.geometry("1000x750")


        self.use_modification_var = tk.BooleanVar(value=True)
        self.nodes = []
        self.connections = []
        self.history = []
        self.deleted_nodes = []
        self.node_id_tracker = 0
        self.active_node = None
        self.active_label_id = None
        self.optimal_route = None
        self.result_text = ""

        self.alpha = self.DEFAULT_PARAMS['alpha']
        self.beta = self.DEFAULT_PARAMS['beta']
        self.rho = self.DEFAULT_PARAMS['rho']
        self.Q = self.DEFAULT_PARAMS['q']
        self.iterations = 50
        self.ants = 20

        self._setup_ui()
        self._center_window()

    def _setup_ui(self):
        core_frame = tk.Frame(self.root)
        core_frame.pack(fill="both", expand=True)
        core_frame.grid_columnconfigure(0, weight=1)
        core_frame.grid_columnconfigure(1, weight=1)
        core_frame.grid_rowconfigure(0, weight=1)

        panel_left = tk.Frame(core_frame)
        panel_left.grid(row=0, column=0, sticky="nsew")
        panel_left.grid_rowconfigure(0, weight=1)
        panel_left.grid_rowconfigure(1, weight=1)
        panel_left.grid_columnconfigure(0, weight=1)

        input_section = tk.LabelFrame(panel_left, text="Граф ввода")
        input_section.grid(row=0, column=0, sticky="nsew")
        input_frame = tk.Frame(input_section)
        input_frame.pack(fill="both", expand=True)

        self.input_area = tk.Canvas(input_frame, bg="lightgray", width=500, height=500, scrollregion=(0, 0, 1000, 1000))
        h_scrollbar_input = tk.Scrollbar(input_frame, orient="horizontal", command=self.input_area.xview)
        v_scrollbar_input = tk.Scrollbar(input_frame, orient="vertical", command=self.input_area.yview)
        self.input_area.configure(xscrollcommand=h_scrollbar_input.set, yscrollcommand=v_scrollbar_input.set)

        h_scrollbar_input.pack(side="bottom", fill="x")
        v_scrollbar_input.pack(side="right", fill="y")
        self.input_area.pack(side="left", fill="both", expand=True)

        input_section.grid_rowconfigure(0, weight=1)
        input_section.grid_columnconfigure(0, weight=1)

        output_section = tk.LabelFrame(panel_left, text="Граф результата")
        output_section.grid(row=1, column=0, sticky="nsew")
        output_frame = tk.Frame(output_section)
        output_frame.pack(fill="both", expand=True)

        self.output_area = tk.Canvas(output_frame, bg="lightgray", width=500, height=500, scrollregion=(0, 0, 1000, 1000))
        h_scrollbar_output = tk.Scrollbar(output_frame, orient="horizontal", command=self.output_area.xview)
        v_scrollbar_output = tk.Scrollbar(output_frame, orient="vertical", command=self.output_area.yview)
        self.output_area.configure(xscrollcommand=h_scrollbar_output.set, yscrollcommand=v_scrollbar_output.set)

        h_scrollbar_output.pack(side="bottom", fill="x")
        v_scrollbar_output.pack(side="right", fill="y")
        self.output_area.pack(side="left", fill="both", expand=True)

        output_section.grid_rowconfigure(0, weight=1)
        output_section.grid_columnconfigure(0, weight=1)

        panel_right = tk.Frame(core_frame)
        panel_right.grid(row=0, column=1, sticky="nsew")
        panel_right.grid_rowconfigure(0, weight=2)
        panel_right.grid_rowconfigure(1, weight=1)
        panel_right.grid_rowconfigure(2, weight=1)
        panel_right.grid_columnconfigure(0, weight=1)

        edges_section = tk.LabelFrame(panel_right, text="Список рёбер")
        edges_section.grid(row=0, column=0, sticky="nsew")
        self.edge_table = ttk.Treeview(edges_section, columns=("From", "To", "Cost"), show="headings")
        self.edge_table.heading("From", text="Начало")
        self.edge_table.heading("To", text="Конец")
        self.edge_table.heading("Cost", text="Вес")
        self.edge_table.pack(fill="both", expand=True)
        for col in ("From", "To", "Cost"):
            self.edge_table.column(col, width=30, stretch=True)

        control_section = tk.LabelFrame(panel_right, text="Управление")
        control_section.grid(row=1, column=0, sticky="nsew")
        ttk.Checkbutton(control_section, text="Использовать модификацию", variable=self.use_modification_var).pack(fill="x", expand=True)
        tk.Button(control_section, text="Найти маршрут", command=self._solve_tsp).pack(fill="x", expand=True)
        tk.Button(control_section, text="Назад", command=self._revert_last_step).pack(fill="x", expand=True)
        tk.Button(control_section, text="Сбросить", command=self._reset_all).pack(fill="x", expand=True)
        tk.Button(control_section, text="Загрузить граф", command=self._load_graph).pack(fill="x", expand=True)
        tk.Button(control_section, text="Скачать результат", command=self._save_result).pack(fill="x", expand=True)

        self.result_container = tk.LabelFrame(panel_right, text="Результат")
        self.result_container.grid(row=2, column=0, sticky="nsew")
        tk.Label(self.result_container, text="Ожидание").pack(fill="both", expand=True)

        self.input_area.bind("<Button-1>", self._place_node)
        self.input_area.bind("<Button-3>", self._pick_node_for_link)

    def _center_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _load_graph(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path:
            return

        self._reset_all()  

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                nodes_section = False  
                node_ids = set()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('# Nodes'):
                        nodes_section = True
                        continue
                    if line.startswith('# Edges'):
                        nodes_section = False
                        continue
                    if not nodes_section:
                        continue 

                    parts = line.split(',')
                    if len(parts) != 3:
                        continue 

                    try:
                        node_id, x, y = map(int, parts)
                        if node_id in node_ids:
                            raise ValueError(f"Дублирующийся ID узла: {node_id}")
                        node_ids.add(node_id)
                        new_node = {"id": node_id, "x_coord": x, "y_coord": y}
                        self.nodes.append(new_node)
                        self.input_area.create_oval(x - self.NODE_SIZE, y - self.NODE_SIZE,
                                                    x + self.NODE_SIZE, y + self.NODE_SIZE, fill="green")
                        self.input_area.create_text(x, y, text=str(node_id), fill="white")
                        self.history.append(("node_added", new_node, None, None))
                        self.node_id_tracker = max(self.node_id_tracker, node_id)
                    except ValueError:
                        continue 

                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.startswith('# Nodes'):
                        continue  

                    parts = line.split(',')
                    if len(parts) != 3:
                        continue

                    try:
                        from_id, to_id, weight = map(int, parts)
                        if from_id not in node_ids or to_id not in node_ids:
                            raise ValueError(f"Узел {from_id} или {to_id} не найден")
                        start_node = next(n for n in self.nodes if n["id"] == from_id)
                        end_node = next(n for n in self.nodes if n["id"] == to_id)
                        link_id = self._render_directed_link(start_node, end_node)
                        self.connections.append((from_id, to_id, weight, link_id))
                        self.edge_table.insert("", "end", values=(from_id, to_id, weight))
                        self.history.append(("link_added", (from_id, to_id, weight, link_id)))
                    except ValueError:
                        continue

                if not self.nodes:
                    raise ValueError("Файл не содержит узлов")
                
                max_x = max(node["x_coord"] for node in self.nodes) + self.NODE_SIZE
                max_y = max(node["y_coord"] for node in self.nodes) + self.NODE_SIZE
                self.input_area.configure(scrollregion=(0, 0, max_x, max_y))
                
                messagebox.showinfo("Успех", "Граф загружен!")
        except Exception as e:
            self._reset_all()
            messagebox.showerror("Ошибка", f"Не удалось загрузить граф: {str(e)}")

    def _place_node(self, event):
        pos_x, pos_y = event.x, event.y
        for node in self.nodes:
            dist = math.sqrt((pos_x - node["x_coord"])**2 + (pos_y - node["y_coord"])**2)
            if dist < self.MIN_SPACING:
                return

        if self.deleted_nodes:
            node_id = min(self.deleted_nodes)
            self.deleted_nodes.remove(node_id)
        else:
            self.node_id_tracker += 1
            node_id = self.node_id_tracker

        new_node = {"id": node_id, "x_coord": pos_x, "y_coord": pos_y}
        self.nodes.append(new_node)

        node_shape = self.input_area.create_oval(pos_x - self.NODE_SIZE, pos_y - self.NODE_SIZE, 
                                                 pos_x + self.NODE_SIZE, pos_y + self.NODE_SIZE, fill="green")
        label_id = self.input_area.create_text(pos_x, pos_y, text=str(node_id), fill="white")

        self.history.append(("node_added", new_node, node_shape, label_id))

    def _pick_node_for_link(self, event):
        pos_x, pos_y = event.x, event.y
        for node in self.nodes:
            dist = math.sqrt((pos_x - node["x_coord"])**2 + (pos_y - node["y_coord"])**2)
            if dist < self.SELECTION_RADIUS:
                if self.active_node is None:
                    self.active_node = node
                    self.active_label_id = self.input_area.create_text(node["x_coord"], node["y_coord"] - 30, 
                                                                      text=f"Выбран узел: {node['id']}", fill="green")
                    return
                else:
                    if self.active_node != node:
                        existing_link_idx = -1
                        for idx, link in enumerate(self.connections):
                            if link[0] == self.active_node["id"] and link[1] == node["id"]:
                                existing_link_idx = idx
                                break

                        if existing_link_idx != -1:
                            old_weight = self.connections[existing_link_idx][2]
                            new_weight = askinteger("Вес связи", 
                                                    f"Связь {self.active_node['id']} -> {node['id']} существует. Вес: {old_weight}\nНовый вес:")
                            if new_weight is not None:
                                old_link = self.connections[existing_link_idx]
                                self.connections[existing_link_idx] = (self.active_node["id"], node["id"], new_weight, old_link[3])
                                self.edge_table.item(self.edge_table.get_children()[existing_link_idx], 
                                                     values=(self.active_node["id"], node["id"], new_weight))
                                self.history.append(("link_updated", old_link, self.connections[existing_link_idx]))

                            self.input_area.delete(self.active_label_id)
                            self.active_node = None
                            self.active_label_id = None
                            return

                        weight = askinteger("Вес связи", f"Укажите вес для связи {self.active_node['id']} -> {node['id']}:")
                        if weight is None:
                            return

                        link_id = self._render_directed_link(self.active_node, node)
                        if link_id:
                            self.connections.append((self.active_node["id"], node["id"], weight, link_id))
                            self.edge_table.insert("", "end", values=(self.active_node["id"], node["id"], weight))
                            self.history.append(("link_added", (self.active_node["id"], node["id"], weight, link_id)))

                        self.input_area.delete(self.active_label_id)
                        self.active_node = None
                        self.active_label_id = None
                    else:
                        self.input_area.delete(self.active_label_id)
                        self.active_node = None
                        self.active_label_id = None

    def _render_directed_link(self, start, end):
        dx = end["x_coord"] - start["x_coord"]
        dy = end["y_coord"] - start["y_coord"]
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return None
        dx /= length
        dy /= length
        begin_x = start["x_coord"] + dx * self.NODE_SIZE
        begin_y = start["y_coord"] + dy * self.NODE_SIZE
        finish_x = end["x_coord"] - dx * self.NODE_SIZE
        finish_y = end["y_coord"] - dy * self.NODE_SIZE
        link_id = self.input_area.create_line(begin_x, begin_y, finish_x, finish_y, 
                                              arrow=tk.LAST, fill="black", width=2, 
                                              arrowshape=(8, 8, 4))
        return link_id

    def _display_optimal_route(self, route, graph_data):
        self.output_area.delete("all")
        for node_id in route:
            node = next(n for n in self.nodes if n["id"] == node_id)
            self.output_area.create_oval(node["x_coord"] - self.NODE_SIZE, node["y_coord"] - self.NODE_SIZE, 
                                        node["x_coord"] + self.NODE_SIZE, node["y_coord"] + self.NODE_SIZE, fill="green")
            self.output_area.create_text(node["x_coord"], node["y_coord"], text=str(node_id), fill="white")
        for i in range(len(route) - 1):
            start_node = next(n for n in self.nodes if n["id"] == route[i])
            end_node = next(n for n in self.nodes if n["id"] == route[i + 1])
            self._render_link_on_output(start_node, end_node)
        start_node = next(n for n in self.nodes if n["id"] == route[-1])
        end_node = next(n for n in self.nodes if n["id"] == route[0])
        self._render_link_on_output(start_node, end_node)
        
        max_x = max(node["x_coord"] for node in self.nodes if node["id"] in route) + self.NODE_SIZE
        max_y = max(node["y_coord"] for node in self.nodes if node["id"] in route) + self.NODE_SIZE
        self.output_area.configure(scrollregion=(0, 0, max_x, max_y))

    def _render_link_on_output(self, start, end):
        dx = end["x_coord"] - start["x_coord"]
        dy = end["y_coord"] - start["y_coord"]
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return
        dx /= length
        dy /= length
        begin_x = start["x_coord"] + dx * self.NODE_SIZE
        begin_y = start["y_coord"] + dy * self.NODE_SIZE
        finish_x = end["x_coord"] - dx * self.NODE_SIZE
        finish_y = end["y_coord"] - dy * self.NODE_SIZE
        self.output_area.create_line(begin_x, begin_y, finish_x, finish_y, 
                                     arrow=tk.LAST, fill="red", width=2, 
                                     arrowshape=(8, 8, 4))
        
    def _solve_tsp(self):
        """Решает TSP с использованием муравьиного алгоритма."""
        pass


    def _save_result(self):
        if not self.optimal_route and "Маршрут не найден" in self.result_text:
            messagebox.showwarning("Предупреждение", "Нет оптимального маршрута для сохранения!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                                                title="Сохранить результат")
        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding="utf-8") as file:
                file.write("# Nodes\n")
                if self.optimal_route:
                    for node_id in self.optimal_route:
                        node = next(n for n in self.nodes if n["id"] == node_id)
                        file.write(f"{node['id']},{node['x_coord']},{node['y_coord']}\n")
                
                file.write("# Edges\n")
                if self.optimal_route:
                    for i in range(len(self.optimal_route) - 1):
                        start_id = self.optimal_route[i]
                        end_id = self.optimal_route[i + 1]
                        weight = next((link[2] for link in self.connections if link[0] == start_id and link[1] == end_id), 0)
                        file.write(f"{start_id},{end_id},{weight}\n")
                    start_id = self.optimal_route[-1]
                    end_id = self.optimal_route[0]
                    weight = next((link[2] for link in self.connections if link[0] == start_id and link[1] == end_id), 0)
                    file.write(f"{start_id},{end_id},{weight}\n")

                file.write("# Result Info\n")
                file.write(self.result_text)

            messagebox.showinfo("Успех", "Результат успешно сохранён!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить результат: {str(e)}")

    def _revert_last_step(self):
        if not self.history:
            return
        last_step = self.history.pop()
        if last_step[0] == "node_added":
            self.deleted_nodes.append(last_step[1]["id"])
            self.nodes.remove(last_step[1])
            self.input_area.delete(last_step[2])
            self.input_area.delete(last_step[3])
            if self.active_node == last_step[1]:
                if self.active_label_id:
                    self.input_area.delete(self.active_label_id)
                self.active_node = None
                self.active_label_id = None
        elif last_step[0] == "link_added":
            link_to_remove = last_step[1]
            self.connections = [link for link in self.connections if not (link[0] == link_to_remove[0] and link[1] == link_to_remove[1])]
            if link_to_remove[3]:
                self.input_area.delete(link_to_remove[3])
            for row in self.edge_table.get_children():
                if self.edge_table.item(row, "values")[:2] == (str(link_to_remove[0]), str(link_to_remove[1])):
                    self.edge_table.delete(row)
                    break
        elif last_step[0] == "link_updated":
            old_link, new_link = last_step[1], last_step[2]
            for i, link in enumerate(self.connections):
                if link[0] == new_link[0] and link[1] == new_link[1]:
                    self.connections[i] = old_link
                    self.edge_table.item(self.edge_table.get_children()[i], values=(old_link[0], old_link[1], old_link[2]))
                    break

    def _reset_all(self):
        self.edge_table.delete(*self.edge_table.get_children())
        self.input_area.delete("all")
        self.output_area.delete("all")
        self.nodes.clear()
        self.connections.clear()
        self.node_id_tracker = 0
        self.active_node = None
        if self.active_label_id:
            self.input_area.delete(self.active_label_id)
            self.active_label_id = None
        self.optimal_route = None
        self.result_text = ""
        for widget in self.result_container.winfo_children():
            widget.destroy()
        tk.Label(self.result_container, text="Очищено").pack(fill="both", expand=True)

if __name__ == "__main__":
    app = tk.Tk()
    tsp_app = TSPApp(app)
    app.mainloop()  # type: ignore
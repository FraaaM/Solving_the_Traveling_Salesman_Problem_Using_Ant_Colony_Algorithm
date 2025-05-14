import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import math
from tkinter.simpledialog import askinteger, askfloat
import time
import random
import json
from collections import defaultdict

class TSPApp:
    NODE_SIZE = 12
    MIN_SPACING = 50
    SELECTION_RADIUS = 17
    DEFAULT_PARAMS = {'alpha': 1.0, 'beta': 2.0, 'rho': 0.3, 'q': 100}

    def __init__(self, root):
        self.root = root
        self.root.title("Алгоритм колонии муравьев для TSP")
        self.root.geometry("1200x750")
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.nodes = []
        self.connections = []
        self.history = []
        self.deleted_nodes = []
        self.node_id_tracker = 0
        self.active_node = None
        self.active_label_id = None
        self.optimal_route = None
        self.result_text = ""
        self.running = False
        
        self.alpha = self.DEFAULT_PARAMS['alpha']
        self.beta = self.DEFAULT_PARAMS['beta']
        self.rho = self.DEFAULT_PARAMS['rho']
        self.Q = self.DEFAULT_PARAMS['q']
        self.iterations = 40
        self.ants = 20
        self.use_modification = tk.BooleanVar(value=True)

        self._setup_ui()
        self._center_window()

    def _setup_ui(self):
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame)
        main_frame.add(left_panel, weight=3)
        
        self._create_canvas_frame(left_panel, "Исходный Граф", 'input_area')
        self._create_canvas_frame(left_panel, "Оптимальный Путь", 'output_area')

        right_panel = ttk.Frame(main_frame)
        main_frame.add(right_panel, weight=1)

        top_right_frame = ttk.Frame(right_panel)
        top_right_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        params_frame = ttk.LabelFrame(top_right_frame, text="Параметры Алгоритма")
        params_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        self._create_param_entry(params_frame, "Alpha (влияние феромонов):", 'alpha', 0)
        self._create_param_entry(params_frame, "Beta (влияние расстояния):", 'beta', 1)
        self._create_param_entry(params_frame, "Rho (испарение):", 'rho', 2)
        self._create_param_entry(params_frame, "Q (количество феромона):", 'Q', 3)
        self._create_param_entry(params_frame, "Итерации:", 'iterations', 4, int)
        self._create_param_entry(params_frame, "Муравьи:", 'ants', 5, int)
        
        ttk.Checkbutton(params_frame, text="Использовать модификацию", variable=self.use_modification).grid(row=6, column=0, sticky='w', pady=2)

        edges_frame = ttk.LabelFrame(top_right_frame, text="Список Ребер")
        edges_frame.grid(row=0, column=1, sticky='nsew', padx=1, pady=1, ipadx=1, ipady=1)
        
        self.edge_table = ttk.Treeview(edges_frame, columns=("From", "To", "Cost"), show="headings", height=12)
        vsb = ttk.Scrollbar(edges_frame, orient="vertical", command=self.edge_table.yview)
        hsb = ttk.Scrollbar(edges_frame, orient="horizontal", command=self.edge_table.xview)
        self.edge_table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        for col, width in [("From", 50), ("To", 50), ("Cost", 50)]:
            self.edge_table.heading(col, text=col)
            self.edge_table.column(col, width=width, anchor=tk.CENTER)
            
        self.edge_table.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        top_right_frame.columnconfigure(0, weight=1)
        top_right_frame.columnconfigure(1, weight=3)
        top_right_frame.rowconfigure(0, weight=1)
        edges_frame.columnconfigure(0, weight=1)
        edges_frame.rowconfigure(0, weight=1)


        ctrl_frame = ttk.LabelFrame(right_panel, text="Управление")
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        buttons = [
            ("Найти Решение", self._solve_tsp),
            ("Загрузить Граф", self._load_graph),
            ("Остановить", self._stop_calculation),
            ("Сохранить Результат", self._save_result),
            ("Назад", self._revert_last_step),
            ("Сбросить", self._reset_all)
        ]
        
        for i, (text, cmd) in enumerate(buttons):
            btn = ttk.Button(ctrl_frame, text=text, command=cmd)
            btn.grid(row=i//2, column=i%2, sticky='nsew', padx=0, pady=0)
        
        for i in range(2):
            ctrl_frame.columnconfigure(i, weight=1)
        for i in range(2):
            ctrl_frame.rowconfigure(i, weight=1)

        self.result_area = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, height=8)
        self.result_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.input_area.bind("<Button-1>", self._place_node)
        self.input_area.bind("<Button-3>", self._pick_node_for_link)

    def _create_canvas_frame(self, parent, title, canvas_var):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(frame, bg="white", scrollregion=(0, 0, 2000, 2000))
        h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        v_scroll = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        setattr(self, canvas_var, canvas)

    def _create_param_entry(self, parent, label, param, row, validate=float):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky='ew', pady=2)
        
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, width=8)
        entry.pack(side=tk.RIGHT)
        entry.insert(0, str(getattr(self, param)))
        
        entry.bind("<FocusOut>", lambda e: self._validate_param(entry, param, validate))

    def _validate_param(self, entry, param, validate):
        try:
            value = validate(entry.get())
            if param in ['alpha', 'beta', 'rho'] and (value <= 0 or value > 5):
                raise ValueError("Значение должно быть от 0 до 5")
            setattr(self, param, value)
            entry.config(foreground='black')
        except ValueError:
            entry.config(foreground='red')
            messagebox.showerror("Ошибка", f"Недопустимое значение для {param}")

    def _center_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _place_node(self, event):
        if self.running:
            return
            
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

        new_node = {
            "id": node_id, 
            "x_coord": pos_x, 
            "y_coord": pos_y,
            "shape": None,
            "label": None
        }
        self.nodes.append(new_node)

        new_node["shape"] = self.input_area.create_oval(
            pos_x - self.NODE_SIZE, pos_y - self.NODE_SIZE,
            pos_x + self.NODE_SIZE, pos_y + self.NODE_SIZE,
            fill="green", outline="black", tags=f"node_{node_id}"
        )
        new_node["label"] = self.input_area.create_text(
            pos_x, pos_y, 
            text=str(node_id), 
            fill="white", 
            tags=f"label_{node_id}"
        )

        self.history.append(("node_added", new_node.copy()))

    def _pick_node_for_link(self, event):
        if self.running:
            return
            
        pos_x, pos_y = event.x, event.y
        for node in self.nodes:
            dist = math.sqrt((pos_x - node["x_coord"])**2 + (pos_y - node["y_coord"])**2)
            if dist < self.SELECTION_RADIUS:
                if self.active_node is None:
                    self.active_node = node
                    self.active_label_id = self.input_area.create_text(
                        node["x_coord"], node["y_coord"] - 30,
                        text=f"Selected: {node['id']}", fill="#2196F3",
                        tags="selection_label"
                    )
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
                            new_weight = askinteger("Edge Weight", 
                                                   f"Existing edge {self.active_node['id']} -> {node['id']}\nNew weight:",
                                                   initialvalue=old_weight)
                            if new_weight is not None:
                                old_link = self.connections[existing_link_idx]
                                self.connections[existing_link_idx] = (
                                    self.active_node["id"], 
                                    node["id"], 
                                    new_weight, 
                                    old_link[3]
                                )
                                self.edge_table.item(
                                    self.edge_table.get_children()[existing_link_idx], 
                                    values=(self.active_node["id"], node["id"], new_weight)
                                )
                                self.history.append(("link_updated", old_link, self.connections[existing_link_idx]))

                            self.input_area.delete("selection_label")
                            self.active_node = None
                            self.active_label_id = None
                            return

                        weight = askinteger("Edge Weight", f"Weight for {self.active_node['id']} -> {node['id']}:")
                        if weight is None:
                            return

                        link_id = self._render_directed_link(self.active_node, node)
                        if link_id:
                            self.connections.append((
                                self.active_node["id"], 
                                node["id"], 
                                weight, 
                                link_id
                            ))
                            self.edge_table.insert(
                                "", "end", 
                                values=(self.active_node["id"], node["id"], weight)
                            )
                            self.history.append((
                                "link_added", 
                                (self.active_node["id"], node["id"], weight, link_id)
                            ))

                    self.input_area.delete("selection_label")
                    self.active_node = None
                    self.active_label_id = None

    def _update_link_position(self, link_id, start_node, end_node):
        dx = end_node["x_coord"] - start_node["x_coord"]
        dy = end_node["y_coord"] - start_node["y_coord"]
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return
        dx /= length
        dy /= length
        begin_x = start_node["x_coord"] + dx * self.NODE_SIZE
        begin_y = start_node["y_coord"] + dy * self.NODE_SIZE
        finish_x = end_node["x_coord"] - dx * self.NODE_SIZE
        finish_y = end_node["y_coord"] - dy * self.NODE_SIZE
        self.input_area.coords(link_id, begin_x, begin_y, finish_x, finish_y)

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
        link_id = self.input_area.create_line(
            begin_x, begin_y, finish_x, finish_y,
            arrow=tk.LAST, fill="#607D8B", width=2,
            arrowshape=(8, 8, 4), tags=f"link_{start['id']}_{end['id']}"
        )
        return link_id

    def _solve_tsp(self):
        if self.running:
            return
            
        self.result_area.delete(1.0, tk.END)
        
        if len(self.nodes) < 2:
            self.result_area.insert(tk.END, "Ошибка: недостаточно узлов для расчета")
            return

        graph_data = {node["id"]: {} for node in self.nodes}
        for link in self.connections:
            graph_data[link[0]][link[1]] = link[2]

        if not self._is_connected(graph_data):
            self.result_area.insert(tk.END, "Ошибка: Граф не связан")
            return

        self.running = True
        self.result_area.insert(tk.END, "Расчет оптимального маршрута...\n")
        self.root.update()
        
        try:
            start_time = time.time()
            self.optimal_route, min_total_cost = self.ant_colony_optimization(graph_data)
            execution_time = (time.time() - start_time) * 1000

            if self.optimal_route and min_total_cost != float('inf'):
                route_str = " -> ".join(map(str, self.optimal_route))
                result_text = (
                    f"Оптимальный маршрут:\n{route_str}\n"
                    f"Общая длина пути: {min_total_cost}\n"
                    f"Время исполнения: {execution_time:.2f} ms\n"
                    f"Параметры: α={self.alpha}, β={self.beta}, ρ={self.rho}, Q={self.Q}\n"
                    f"Итерации: {self.iterations}, Муравьи: {self.ants}"
                )
                self._display_optimal_route(self.optimal_route)
            else:
                result_text = f"Не найден допустимый маршрут\nВремя выполнения: {execution_time:.2f} ms"

            self.result_area.delete(1.0, tk.END)
            self.result_area.insert(tk.END, result_text)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Расчет не удался: {str(e)}")
        finally:
            self.running = False

    def _stop_calculation(self):
        self.running = False
        self.result_area.insert(tk.END, "\nРасчет остановлен пользователем")

    def _is_connected(self, graph):
        if not graph:
            return False
            
        nodes = list(graph.keys())
        visited = set()
        stack = [nodes[0]]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(graph[node].keys())
        
        return len(visited) == len(nodes)

    def ant_colony_optimization(self, graph_data):
        nodes = list(graph_data.keys())
        if not nodes:
            return None, float('inf')

        num_ants = self.ants
        num_iterations = self.iterations
        pheromone = self._initialize_pheromones(graph_data)

        best_route = None
        best_length = float('inf')
        progress_interval = max(1, num_iterations // 10)

        for iteration in range(num_iterations):
            if not self.running:
                break
                
            all_routes, all_lengths = self._construct_routes(graph_data, pheromone, num_ants)
            self._update_pheromones(pheromone, all_routes, all_lengths)

            current_best = min(all_lengths) if all_lengths else float('inf')
            if current_best < best_length:
                best_route = all_routes[all_lengths.index(current_best)]
                best_length = current_best
                
                if iteration % progress_interval == 0:
                    self.result_area.insert(
                        tk.END, 
                        f"Iteration {iteration+1}/{num_iterations}: Best cost = {best_length}\n"
                    )
                    self.root.update()

        return best_route, best_length

    def _initialize_pheromones(self, graph_data):
        pheromone = defaultdict(dict)
        for i in graph_data:
            for j in graph_data[i]:
                pheromone[i][j] = 1.0
        return pheromone

    def _construct_routes(self, graph_data, pheromone, num_ants):
        all_routes = []
        all_lengths = []
        nodes = list(graph_data.keys())

        for _ in range(num_ants):
            if not self.running:
                break
                
            current_node = random.choice(nodes)
            visited = [current_node]
            length = 0

            while len(visited) < len(nodes):
                next_nodes = [j for j in graph_data[current_node] if j not in visited]
                if not next_nodes:
                    break

                probabilities = []
                total = 0.0
                for j in next_nodes:
                    tau = pheromone[current_node].get(j, 1e-6)
                    eta = 1.0 / graph_data[current_node][j]
                    prob = (tau ** self.alpha) * (eta ** self.beta)
                    probabilities.append(prob)
                    total += prob

                if total == 0:
                    next_node = random.choice(next_nodes)
                else:
                    probabilities = [p / total for p in probabilities]
                    next_node = random.choices(next_nodes, weights=probabilities, k=1)[0]

                length += graph_data[current_node][next_node]
                visited.append(next_node)
                current_node = next_node

            if len(visited) == len(nodes) and visited[0] in graph_data[visited[-1]]:
                length += graph_data[visited[-1]][visited[0]]
                visited.append(visited[0])
                all_routes.append(visited)
                all_lengths.append(length)

        return all_routes, all_lengths

    def _update_pheromones(self, pheromone, all_routes, all_lengths):
        for i in pheromone:
            for j in pheromone[i]:
                pheromone[i][j] *= (1 - self.rho)

        for route, length in zip(all_routes, all_lengths):
            if length == 0:
                continue
            delta = self.Q / length
            for i in range(len(route)-1):
                if route[i+1] in pheromone[route[i]]:
                    pheromone[route[i]][route[i+1]] += delta

            if self.use_modification.get():
                edge_counts = defaultdict(int)
                for i in range(len(route)-1):
                    edge_counts[(route[i], route[i+1])] += 1
                
                for edge, count in edge_counts.items():
                    if edge[1] in pheromone[edge[0]]:
                        pheromone[edge[0]][edge[1]] += 0.1 * self.Q * count

    def _display_optimal_route(self, route):
        self.output_area.delete("all")
        colors = ['#%02x%02x%02x' % (r, g, 150) for r, g in zip(
            range(50, 255, 25), range(100, 0, -25))]
        
        node_positions = {}
        for idx, node_id in enumerate(route[:-1]): 
            node = next(n for n in self.nodes if n["id"] == node_id)
            node_positions[node_id] = (node["x_coord"], node["y_coord"])
            color = colors[idx % len(colors)]
            self.output_area.create_oval(
                node["x_coord"] - self.NODE_SIZE, node["y_coord"] - self.NODE_SIZE,
                node["x_coord"] + self.NODE_SIZE, node["y_coord"] + self.NODE_SIZE, 
                fill=color, outline='black', tags=f"route_node_{node_id}"
            )
            self.output_area.create_text(
                node["x_coord"], node["y_coord"], 
                text=str(node_id), fill='white', tags=f"route_label_{node_id}"
            )

        for i in range(len(route)-1):
            start_id = route[i]
            end_id = route[i+1]
            self._render_route_link(start_id, end_id, node_positions, color="#FF5722")

    def _render_route_link(self, start_id, end_id, node_positions, color):
        start_x, start_y = node_positions[start_id]
        end_x, end_y = node_positions[end_id]
        
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return
            
        dx /= length
        dy /= length
        begin_x = start_x + dx * self.NODE_SIZE
        begin_y = start_y + dy * self.NODE_SIZE
        finish_x = end_x - dx * self.NODE_SIZE
        finish_y = end_y - dy * self.NODE_SIZE
        
        self.output_area.create_line(
            begin_x, begin_y, finish_x, finish_y,
            arrow=tk.LAST, fill=color, width=3,
            arrowshape=(10, 10, 5), tags=f"route_link_{start_id}_{end_id}"
        )

    def _load_graph(self):
        if self.running:
            return
            
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not file_path:
            return

        # # Сброс текущего графа
        # self.edge_table.delete(*self.edge_table.get_children())
        # self.input_area.delete("all")
        # self.nodes.clear()
        # self.connections.clear()
        # self.node_id_tracker = 0
        self._reset_all()
        
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                nodes_section = False
                node_ids = set()
                max_x = max_y = 0
                
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        nodes_section = line.startswith('# Nodes')
                        continue
                        
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) != 3:
                        continue
                        
                    try:
                        if nodes_section:
                            node_id, x, y = map(int, parts)
                            if node_id in node_ids:
                                continue
                                
                            node = {
                                "id": node_id,
                                "x_coord": x,
                                "y_coord": y,
                                "shape": self.input_area.create_oval(
                                    x - self.NODE_SIZE, y - self.NODE_SIZE,
                                    x + self.NODE_SIZE, y + self.NODE_SIZE,
                                    fill="#4CAF50", outline="black", tags=f"node_{node_id}"
                                ),
                                "label": self.input_area.create_text(
                                    x, y, text=str(node_id), fill="white", tags=f"label_{node_id}"
                                )
                            }
                            self.nodes.append(node)
                            node_ids.add(node_id)
                            self.node_id_tracker = max(self.node_id_tracker, node_id)
                            max_x, max_y = max(max_x, x), max(max_y, y)
                        else:
                            from_id, to_id, weight = map(int, parts)
                            if from_id not in node_ids or to_id not in node_ids:
                                continue
                                
                            start_node = next(n for n in self.nodes if n["id"] == from_id)
                            end_node = next(n for n in self.nodes if n["id"] == to_id)
                            
                            dx = end_node["x_coord"] - start_node["x_coord"]
                            dy = end_node["y_coord"] - start_node["y_coord"]
                            length = math.sqrt(dx**2 + dy**2)
                            if length == 0:
                                continue
                                
                            dx, dy = dx/length, dy/length
                            x1 = start_node["x_coord"] + dx * self.NODE_SIZE
                            y1 = start_node["y_coord"] + dy * self.NODE_SIZE
                            x2 = end_node["x_coord"] - dx * self.NODE_SIZE
                            y2 = end_node["y_coord"] - dy * self.NODE_SIZE
                            
                            link_id = self.input_area.create_line(
                                x1, y1, x2, y2, arrow=tk.LAST, 
                                fill="#607D8B", width=2, arrowshape=(8, 8, 4),
                                tags=f"link_{from_id}_{to_id}"
                            )
                            
                            self.connections.append((from_id, to_id, weight, link_id))
                            self.edge_table.insert("", "end", values=(from_id, to_id, weight))
                    
                    except (ValueError, StopIteration):
                        continue
                        
            if not self.nodes:
                raise ValueError("Файл не содержит корректных узлов")
                
            self.input_area.configure(
                scrollregion=(0, 0, 
                            max_x + self.NODE_SIZE * 2, 
                            max_y + self.NODE_SIZE * 2)
            )
            
            messagebox.showinfo("Успех", "Граф успешно загружен")
            
        except Exception as e:
            self._reset_all()
            messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")

    def _save_result(self):
        if self.running:
            return
            
        if not self.optimal_route or "Нет допустимого маршрута" in self.result_area.get(1.0, tk.END):
            messagebox.showwarning("Предупреждение", "Нет оптимального маршрута для сохранения!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Result"
        )
        if not file_path:
            return

        try:
            if file_path.endswith('.json'):
                self._save_json_result(file_path)
            else:
                self._save_text_result(file_path)
                
            messagebox.showinfo("Успех", "Результат сохранен!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить результат: {str(e)}")

    def _save_json_result(self, file_path):
        result = {
            "parameters": {
                "alpha": self.alpha,
                "beta": self.beta,
                "rho": self.rho,
                "Q": self.Q,
                "iterations": self.iterations,
                "ants": self.ants,
                "use_modification": self.use_modification.get()
            },
            "nodes": [],
            "edges": [],
            "route": self.optimal_route,
            "total_cost": None,
            "info": self.result_area.get(1.0, tk.END).strip()
        }

        if self.optimal_route:
            result["total_cost"] = sum(
                next((link[2] for link in self.connections 
                     if link[0] == self.optimal_route[i] and link[1] == self.optimal_route[i+1]), 0)
                for i in range(len(self.optimal_route)-1)
            )

        for node in self.nodes:
            if node["id"] in self.optimal_route:
                result["nodes"].append({
                    "id": node["id"],
                    "x": node["x_coord"],
                    "y": node["y_coord"]
                })

        for i in range(len(self.optimal_route)-1):
            from_id = self.optimal_route[i]
            to_id = self.optimal_route[i+1]
            weight = next((link[2] for link in self.connections 
                         if link[0] == from_id and link[1] == to_id), 0)
            result["edges"].append({
                "from": from_id,
                "to": to_id,
                "weight": weight
            })

        with open(file_path, 'w') as file:
            json.dump(result, file, indent=2)

    def _save_text_result(self, file_path):
        with open(file_path, 'w',encoding="utf-8") as file:
            file.write("# Result Info\n")
            file.write(self.result_area.get(1.0, tk.END))
            file.write("\n\n# Nodes\n")
            for node in self.nodes:
                if node["id"] in self.optimal_route:
                    file.write(f"{node['id']},{node['x_coord']},{node['y_coord']}\n")
            
            file.write("\n# Edges\n")
            for i in range(len(self.optimal_route)-1):
                from_id = self.optimal_route[i]
                to_id = self.optimal_route[i+1]
                weight = next((link[2] for link in self.connections 
                             if link[0] == from_id and link[1] == to_id), 0)
                file.write(f"{from_id},{to_id},{weight}\n")

    def _revert_last_step(self):
        if self.running or not self.history:
            return
            
        last_step = self.history.pop()
        if last_step[0] == "node_added":
            node = last_step[1]
            self.deleted_nodes.append(node["id"])
            self.nodes.remove(next(n for n in self.nodes if n["id"] == node["id"]))
            self.input_area.delete(f"node_{node['id']}")
            self.input_area.delete(f"label_{node['id']}")
            
            for conn in self.connections[:]:
                if conn[0] == node["id"] or conn[1] == node["id"]:
                    self.input_area.delete(conn[3])
                    self.connections.remove(conn)
                    for row in self.edge_table.get_children():
                        if self.edge_table.item(row, "values")[:2] == (conn[0], conn[1]):
                            self.edge_table.delete(row)
                            break
            
        elif last_step[0] == "link_added":
            link = last_step[1]
            self.connections.remove(next(
                c for c in self.connections 
                if c[0] == link[0] and c[1] == link[1]
            ))
            self.input_area.delete(link[3])
            for row in self.edge_table.get_children():
                if self.edge_table.item(row, "values")[:2] == (link[0], link[1]):
                    self.edge_table.delete(row)
                    break
                    
        elif last_step[0] == "link_updated":
            old_link, new_link = last_step[1], last_step[2]
            for i, conn in enumerate(self.connections):
                if conn[0] == new_link[0] and conn[1] == new_link[1]:
                    self.connections[i] = old_link
                    self.edge_table.item(
                        self.edge_table.get_children()[i], 
                        values=(old_link[0], old_link[1], old_link[2])
                    )
                    break
                    
        elif last_step[0] == "node_moved":
            node = last_step[1]
            current_node = next(n for n in self.nodes if n["id"] == node["id"])
            current_node["x_coord"] = node["x_coord"]
            current_node["y_coord"] = node["y_coord"]
            
            self.input_area.coords(
                current_node["shape"],
                node["x_coord"] - self.NODE_SIZE,
                node["y_coord"] - self.NODE_SIZE,
                node["x_coord"] + self.NODE_SIZE,
                node["y_coord"] + self.NODE_SIZE
            )
            self.input_area.coords(current_node["label"], node["x_coord"], node["y_coord"])
            
            for conn in self.connections:
                if conn[0] == node["id"] or conn[1] == node["id"]:
                    start_node = next(n for n in self.nodes if n["id"] == conn[0])
                    end_node = next(n for n in self.nodes if n["id"] == conn[1])
                    self._update_link_position(conn[3], start_node, end_node)

    def _reset_all(self):
        if self.running:
            return
            
        self.edge_table.delete(*self.edge_table.get_children())
        self.input_area.delete("all")
        self.output_area.delete("all")
        self.nodes.clear()
        self.connections.clear()
        self.history.clear()
        self.deleted_nodes.clear()
        self.node_id_tracker = 0
        self.active_node = None
        if self.active_label_id:
            self.input_area.delete(self.active_label_id)
            self.active_label_id = None
        self.optimal_route = None
        self.result_area.delete(1.0, tk.END)
        self.result_area.insert(tk.END, "Готово к новому расчету")

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()
import random

# Генерация узлов
nodes = []
for i in range(1, 41):
    x = random.randint(50, 1450)
    y = random.randint(50, 1450)
    nodes.append((i, x, y))

# Генерация рёбер (около 80% плотности)
edges = []
for i in range(1, 41):
    for j in range(1, 41):
        if i != j and random.random() < 0.85:  # 85% вероятность связи
            weight = random.randint(1, 100)
            edges.append((i, j, weight))

# Запись в файл
with open("graph_40_nodes_dense.txt", "w") as f:
    f.write("# Nodes\n")
    for node in nodes:
        f.write(f"{node[0]},{node[1]},{node[2]}\n")
    f.write("# Edges\n")
    for edge in edges:
        f.write(f"{edge[0]},{edge[1]},{edge[2]}\n")

print(f"Создан граф с {len(nodes)} узлами и {len(edges)} рёбрами")
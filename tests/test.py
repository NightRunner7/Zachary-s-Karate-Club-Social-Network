# --------------------------- IMPORT ---------------------------
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

# --------------------------- CODE --------------------------- #
# --- flags
makePlot = False
# --- Grabbing this dataset is convenient through networkx
G = nx.karate_club_graph()
print("G:", G)
print("G.nodes", G.nodes)
print("G.nodes.data()", G.nodes.data())
print("G.edges", G.edges)

print("G.edges.data()", G.edges.data())

print("G.nodes[2]", G.nodes[2])
# print("G.edges[2]", G.edges[0])
print("G.adj[0]", G.adj[0])
ngb_list = list(G.adj[0])

print("ngb_list:", ngb_list)

# --- spring layout
pos = nx.spring_layout(G)
print("pos:", pos)

# --- VISUALIZATION
# color the nodes by faction
color = []
for node in G.nodes():
    if G.nodes.data()[node]["club"] == "Mr. Hi":
        color.append("C0")
    else:
        color.append("C1")

if makePlot is True:
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis("equal")
    nx.draw_circular(
        G, with_labels=True, node_color=color, ax=ax, font_color="white", node_size=1000
    )
    ax.set_title("Zachary's Karate Club\nCircular Network Plot", fontsize=20)

    # legend
    john_a_legend = Line2D(
        [],
        [],
        markerfacecolor="C1",
        markeredgecolor="C1",
        marker="o",
        linestyle="None",
        markersize=10,
    )

    mr_hi_legend = Line2D(
        [],
        [],
        markerfacecolor="C0",
        markeredgecolor="C0",
        marker="o",
        linestyle="None",
        markersize=10,
    )

    ax.legend(
        [mr_hi_legend, john_a_legend],
        ["Mr. Hi", "John A."],
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Faction",
    )
    plt.show()

for i, j in G.edges:
    print("i:", i, ", j:", j)
    G.edges[i, j]['weight'] = G.edges[i, j]['weight']

print("-----------------------------------")
G.remove_edges_from(G.edges)

for i in G.nodes():
    for j in G.nodes():
        G.add_edge(i, j)
        G.edges[i, j]['weight'] = 0.5

print("G.edges.data()", G.edges.data())

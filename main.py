from typing import List, Tuple, Iterable, Callable
import random
import os
import json

import numpy as np
import networkx as nx
import requests
import asyncio
import aiohttp
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

SystemProto = Tuple[int, str, float, List[int], int]  # Might work better as a named tuple
JITA = 30000142


"""Get a system from ESI."""
async def get_system(system_id: int, session: aiohttp.ClientSession) -> SystemProto:
    try:
        url = f"https://esi.evetech.net/latest/universe/systems/{system_id}/?datasource=tranquility"
        async with session.get(url=url) as response:
            system = await response.json()
        system_name = system["name"]
        sec_status = system["security_status"]
        connections = []
        if "stargates" in system:
            for stargate_id in system["stargates"]:
                url = f"https://esi.evetech.net/latest/universe/stargates/{stargate_id}/?datasource=tranquility"
                async with session.get(url=url) as response:
                    stargate = await response.json()        
                connections.append(stargate["destination"]["system_id"])
        n_stations = len(system["stations"]) if "stations" in system else 0
        return system_id, system_name, sec_status, connections, n_stations
    except Exception as e:
        raise ValueError(f"Failed to get system {system_id}", e)


"""Get all systems from ESI."""
async def esi_to_df() -> pd.DataFrame:
    r = requests.get("https://esi.evetech.net/latest/universe/systems/?datasource=tranquility")
    r.raise_for_status()
    all_systems = r.json()
    print(f"Loading {len(all_systems)} systems from ESI...")
    async with aiohttp.ClientSession() as session:
        futures_list = [get_system(system_id, session) for system_id in all_systems]
        data = [await future for future in tqdm(asyncio.as_completed(futures_list), total=len(futures_list))]
    df = pd.DataFrame(data, columns=["system_id", "system_name", "sec_status", "connections", "n_stations"])
    df.set_index("system_id", inplace=True)
    return df


"""Helper method for importing a list of integers from a string, used with pandas converters."""
def str2int_list(s):
    return [int(x) for x in s.strip("[]").split(",") if x]


"""Read the map from a CSV and return a NetworkX graph object."""
def eve_map_graph(
        include_hs = True, 
        include_ls = True, 
        include_ns = True, 
        include_jspace = True, 
        include_pochven = True, 
        include_unreachable = False,
        reachable_from = None
    ) -> nx.Graph:
    if not os.path.exists("systems.csv"):
        raise FileNotFoundError("systems.csv not found")
    df = pd.read_csv("systems.csv", index_col="system_id", converters={"connections": str2int_list, "sec_status": lambda x: round(float(x), 2)})
    # Filter out unwanted security statuses
    # Highsec is 0.45 to 1.0, Lowsec is 0.0 to 0.45, Nullsec is -1.0 to 0.0, J-space has a special system id, Pochven is -1.0 exactly
    # TODO some of these are untested
    if not include_unreachable:
        df = df[df.index < 32000000]
    if not include_hs:
        df = df[df["sec_status"] < 0.45]
    if not include_ls:
        df = df[(df["sec_status"] >= 0.45) | (df["sec_status"] < 0.0)]
    if not include_ns:
        df = df[df["sec_status"] >= 0.0]
    if not include_jspace:
        df = df[(df.index >= 32000000) | (df.index < 31000000)]
    if not include_pochven:
        df = df[df["sec_status"] > -1.0]
    df_exploded = df.explode("connections")
    df_exploded = df_exploded[df_exploded["connections"].isin(df.index)]
    df_exploded["system_id"] = df_exploded.index
    G: nx.Graph = nx.from_pandas_edgelist(df_exploded, source="system_id", target="connections")
    df_attrs = df[["system_name", "sec_status", "n_stations"]]
    nx.set_node_attributes(G, df_attrs.to_dict(orient="index"))
    if reachable_from:
        if reachable_from not in G:
            print(G)
            raise ValueError(f"System {reachable_from} not in graph")
        G = G.subgraph(nx.descendants(G, reachable_from) | {reachable_from}).copy()
    return G


"""Get the d-hop closure of a graph."""
def d_closure(G: nx.Graph, d: int) -> nx.Graph:
    G = G.copy()
    for node, ego_graph in [(node, nx.ego_graph(G, node, radius=d)) for node in G.nodes]:
        new_edges = [(node, neighbor) for neighbor in ego_graph.nodes if neighbor != node]
        G.add_edges_from(new_edges)
    return G


"""Approximate the minimum dominating set of a graph."""
def approx_minimum_dominating_set(G: nx.Graph, seed=None) -> set[int]:
    # TODO I bet a pre-sorting of the nodes would be beneficial
    # Right now it's roughly O(n^2) in the worst case, but it could be O(n log n) with a good sort
    all_nodes = set(G.nodes)
    if not seed or seed not in G.nodes:
        seed = random.choice(all_nodes)
    dominating_set = {seed}
    dominated_nodes = set(G[seed])
    uncovered_nodes = all_nodes - dominated_nodes - dominating_set
    while uncovered_nodes:
        G_tmp = G.subgraph(uncovered_nodes)
        most_connected_node = max(G_tmp.nodes, key=lambda node: len(G_tmp[node]) * 1 if G_tmp.nodes[node]["n_stations"] else 0)
        dominating_set.add(most_connected_node)
        dominated_nodes.update(G[most_connected_node])
        uncovered_nodes -= set(G[most_connected_node])
        uncovered_nodes -= {most_connected_node}
    return dominating_set


"""
Get the TSP path and minimum distances to the MDS for a given d-hop dominating set.
Annotate the graph with the distances to the nearest MDS member and the identity of the nearest MDS member.
Return the TSP path, the list of distances, and the annotated graph.
"""
def evaluate_mds(G: nx.Graph, mds: Iterable[int], d: int) -> int:
    G_annotated = G.copy()  # No side-effects please
    # Distribution of distances to nearest member of the MDS
    distances = []
    # BFS and assign a new attribute to each node for what member of the MDS it is closest to
    for i in reversed(range(d + 2)):
        for member in mds:
            G_ego = nx.ego_graph(G_annotated, member, radius=i)
            nx.set_node_attributes(G_annotated, {node: member for node in G_ego.nodes}, name="closest_mds_member")
            nx.set_node_attributes(G_annotated, {node: i for node in G_ego.nodes}, name=f"distance_to_mds_member")
    distances = np.fromiter(nx.get_node_attributes(G_annotated, "distance_to_mds_member").values(), dtype=int)
    if np.max(distances) > d:
        raise ValueError(f"Distance to nearest MDS member exceeds {d}")
    # TSP cycle distance
    tsp_path = nx.approximation.traveling_salesman_problem(G_annotated, nodes=mds, cycle=True)
    descriptive_stats = {
        "d": d,
        "mds_size": len(mds),
        "tsp_path_length": len(tsp_path) - 1,
        "mean_distance_to_mds_member": float(np.mean(distances)),
        "max_distance_to_mds_member": int(np.max(distances)),
        "mds": list(mds),
        # "tsp_path": list(tsp_path),  # Pretty long
    }
    return descriptive_stats, G_annotated


"""Read data from produced CSVs and plot the results.  (It's expensive to re-run the whole thing.)"""
def do_the_plotting(d_min: int, d_max: int):
    d_vals = list(range(d_min, d_max + 1))
    fig1, axes = plt.subplots(len(d_vals), 1, sharex=True, figsize=(6.0, 8.0))
    for ax in axes:
        ax.set_xticks(list(range(0, 13)))
    fig1.suptitle(f"Minimum d-hop dominating set")
    fig1.supylabel("Frequency")
    fig1.supxlabel("Distance to nearest MDS member")

    d_lens = []
    tsp_lens = []
    for d in d_vals:
        df_mds = pd.read_csv(f"mds_{d}.csv")
        with open(f"mds_{d}.json", "r") as f:
            stats = json.load(f)
        distances = df_mds["distance_to_mds_member"]
        d_lens.append(stats["mds_size"])
        tsp_lens.append(stats["tsp_path_length"])

        ax = axes[d - 3]
        ax.set_xlim(-0.5, 12.5)
        ax.hist(distances, bins=np.arange(0, 13, 1) - 0.5)
        ax.axvline(x=np.mean(distances), color="k", linestyle="dashed")
        ax.set_title(f"{d=}", loc="right", y=1.0, pad=-14)

    fig1.savefig("Figure_1.png", )
    fig2, (ax_count, ax_tsp) = plt.subplots(1, 2, figsize=(6.0, 4.8))
    fig2.suptitle("Minimum d-hop dominating set")
    ax_count.plot(d_vals, d_lens)
    ax_count.set_title("Number of nodes in MDS")
    ax_count.set_xlabel("d-hop dominating set")
    ax_count.set_ylabel("Nodes")
    ax_tsp.plot(d_vals, tsp_lens)
    ax_tsp.set_title("TSP circuit length")
    ax_tsp.set_xlabel("d-hop dominating set")
    ax_tsp.set_ylabel("Jumps")
    fig2.savefig("Figure_2.png")
    plt.show()


def main(do_calculation: bool = False, do_plotting: bool = False):
    if not os.path.exists("systems.csv"):
        print("Creating systems.csv...")
        df = asyncio.run(esi_to_df())
        df.to_csv("systems.csv")
    d_min = 3
    d_max = 12
    G = eve_map_graph(
        include_hs=True, 
        include_ls=False, 
        include_ns=False, 
        include_jspace=False, 
        include_pochven=False, 
        reachable_from=JITA
    )
    if do_calculation:
        print("Calculating...")
        for d in range(d_min, d_max + 1):
            G_d = d_closure(G, d)
            mds = approx_minimum_dominating_set(G_d, seed=JITA)
            stats, G_annotated = evaluate_mds(G, mds, d)
            df = pd.DataFrame.from_dict(G_annotated.nodes, orient='index')
            df.to_csv(f"mds_{d}.csv")
            with open(f"mds_{d}.json", "w") as f:
                json.dump(stats, f)
            print(f"{d=}, {stats['mds_size']=}")
    if do_plotting:
        print("Plotting...")
        do_the_plotting(d_min, d_max)




if __name__ == "__main__":
    main(False, True)

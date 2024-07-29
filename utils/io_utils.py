import re

from ast import literal_eval
from .math_utils import get_dis, pnt2line
import os

import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

import networkx as nx

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error)

city_names = ["New York City", "San Francisco", "Washington", "Chicago"]


def parse_edgelist_line(line):
    match = re.match(r"\(([^)]+)\) \(([^)]+)\) (.+)", line)
    if match:
        node1_str, node2_str, attr_str = match.groups()
        node1 = literal_eval(f"({node1_str})")
        node2 = literal_eval(f"({node2_str})")
        attr = ""
        return node1, node2, attr
    else:
        raise ValueError("Line format is incorrect")


def load_access_street_view(city, value_path):
    temp = os.listdir(f"/home/wangb/OpenVIRL/data/{city_names[city]}/")

    ava_indexs = []

    for item in temp:
        ava_indexs.append(item)
    return ava_indexs


def read_images(street_views):
    images = {}
    for idx, item in enumerate(street_views):
        path = f"/home/work/zhangruixing/LLM-Investigator/data/StreetView/images/street_view_{int(item[0] + 1)}.jpg"
        image = Image.open(path).convert('RGB')
        images[item[0] + 1] = image
    return images


def get_strat_point(sub_g):
    node_list = []
    x_min, x_max, y_min, y_max = 123456789, -123456789, 123456789, -123456789
    for node in sub_g.nodes:
        node_list.append(node)
        x_min = min(x_min, node[0])
        x_max = max(x_max, node[0])
        y_min = min(y_min, node[1])
        y_max = max(y_max, node[1])

    real_mid = [[(x_min + x_max) / 2, (y_min + y_max) / 2]]

    ans = -1
    minn_dis = 123456789
    for node in node_list:
        if get_dis(node, real_mid[0]) < minn_dis:
            minn_dis = get_dis(node, real_mid[0])
            ans = node

    return ans


def print_bottom(sub_g, street_views, colors_edge, colors):
    fig, ax = plt.subplots(figsize=(10, 10))

    for idx, edge in enumerate(sub_g.edges):
        node1, node2 = edge
        x1, y1 = node1
        x2, y2 = node2
        plt.plot([x1, x2], [y1, y2], color=colors_edge[idx])

    for idx, item in enumerate(street_views):
        point = item[1], item[2]
        plt.plot(item[2], item[1], "o", color=colors[idx])


def assign_color(street_views):
    # random assign a color to each street view
    colors = []
    for item in street_views:
        colors.append(np.random.rand(3, ))
    return colors


def assign_edge_color(sub_g, street_views, color_node):
    colors = []
    for idx1, edge in enumerate(sub_g.edges):
        minn_dis = 123456789
        pos = 0
        for idx2, item in enumerate(street_views):
            point = (item[2], item[1])
            dist = pnt2line(point, edge[0], edge[1])[0]
            if dist < minn_dis:
                sub_g.edges[edge]["image"] = item[0] + 1
                sub_g.edges[edge]["coord"] = [item[1], item[2]]
                pos = idx2
                minn_dis = dist
        colors.append(color_node[pos])
    return colors


def my_read_edge_list(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            node1, node2, attr = parse_edgelist_line(line)
            G.add_edge(node1, node2)
    return G


def get_graph_and_images(index, city, value_path):
    street_views = index.replace(".edgelist", ".npy")
    sub_g = my_read_edge_list(
        f"/home/work/zhangruixing/LLM-Investigator/data/PBFGraph/{city_names[city]}/{value_path}/{index}")
    street_views = np.load(
        f"/home/work/zhangruixing/LLM-Investigator/data/StreetView/{city_names[city]}/{value_path}/{street_views}")
    colors = assign_color(street_views)
    colors_edge = assign_edge_color(sub_g, street_views, colors)
    images = read_images(street_views)
    start_point = get_strat_point(sub_g)
    print_bottom(sub_g, street_views, colors_edge, colors)

    return sub_g, street_views, images, start_point


def get_graph_and_images_dual(index, city, value_path):
    street_views = index.replace(".edgelist", ".npy")
    sub_g = my_read_edge_list(
        f"/home/work/zhangruixing/LLM-Investigator/data/PBFGraph/{city_names[city]}/{value_path}/{index}")
    street_views = np.load(
        f"/home/work/zhangruixing/LLM-Investigator/data/StreetView/{city_names[city]}/{value_path}/{street_views}")
    colors = assign_color(street_views)
    colors_edge = assign_edge_color(sub_g, street_views, colors)
    images = read_images(street_views)
    start_point = get_strat_point(sub_g)
    print_bottom(sub_g, street_views, colors_edge, colors)

    return sub_g, street_views, images


def read_images_new(street_views):
    images = {}
    for idx, item in enumerate(street_views):
        path = f"/home/work/zhangruixing/LLM-Investigator/data/StreetView/images/street_view_{int(item[0] + 1)}.jpg"
        image = Image.open(path).convert('RGB')
        images[item[0] + 1] = image
    return images


def get_images(index, city, value_path):
    index = int(index)
    root_path = f"/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/images"
    temp = os.listdir(root_path)

    images = []

    for idx, item in enumerate(temp):
        path = f"{root_path}/{item}"
        image = Image.open(path).convert('RGB')
        images.append(image)

    return None, images


def load_task_data(city, value_path):
    if value_path != "Both":
        return np.load(f"/home/wangb/zhangrx/LLMInvestrigator/data/TaskData/{city_names[city]}/{value_path}.npy")
    else:
        temp1 = np.load(f"/home/wangb/zhangrx/LLMInvestrigator/data/TaskData/{city_names[city]}/Carbon.npy")
        temp2 = np.load(f"/home/wangb/zhangrx/LLMInvestrigator/data/TaskData/{city_names[city]}/Population.npy")
        return [temp1, temp2]


# return np.load(f"/home/work/zhangruixing/LLM-Investigator/data/TaskData/{city_names[city]}/{value_path}.npy")


def calc(phase, epoch, all_predicts, all_y, loss):
    metrics = {}
    if loss is not None:
        metrics["loss"] = loss
    metrics["mse"] = mean_squared_error(all_y, all_predicts)
    metrics["r2"] = r2_score(all_y, all_predicts)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)

    print(
        f"{phase} Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )
    return metrics

#!/usr/bin/env python3

import matplotlib.pyplot as pp

decision_node = {'boxstyle': 'sawtooth', 'fc': '0.8'}
leaf_node = {'boxstyle': 'round4', 'fc': '0.8'}
arrow_args = {'arrowstyle': '<-'}


def plot(tree):
    fig = pp.figure(1, facecolor='white')
    fig.clf()
    axprops = {'xticks': [], 'yticks': []}
    ax1 = pp.subplot(111, frameon=False, **axprops)
    num_leafs = get_num_leaf_node(tree)
    depth = get_tree_depth(tree)
    plot_tree(ax1, num_leafs, depth, -0.5/num_leafs, 1.0, tree, (0.5, 1.0), '')
    pp.show()


def plot_node(ax, text, center, parent, node_type):
    ax.annotate(text, xy=parent, xycoords='axes fraction',
                xytext=center, textcoords='axes fraction',
                va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def get_num_leaf_node(tree):
    result = 0
    if type(tree) == str:
        result += 1
    else:
        label = list(tree.keys())[0]
        for subtree in tree[label].values():
            result += get_num_leaf_node(subtree)
    return result


def get_tree_depth(tree):
    if type(tree) == str:
        return 1
    else:
        label = list(tree.keys())[0]
        max_sub_depth = 1
        for subtree in tree[label].values():
            sub_depth = get_tree_depth(subtree)
            if sub_depth > max_sub_depth:
                max_sub_depth = sub_depth

        return max_sub_depth + 1


def plot_mid_text(ax, curr_pt, parent_pt, text):
    mid_pt_x = (parent_pt[0] - curr_pt[0]) / 2 + curr_pt[0]
    mid_pt_y = (parent_pt[1] - curr_pt[1]) / 2 + curr_pt[1]
    ax.text(mid_pt_x, mid_pt_y, text)


def plot_tree(ax, total_w, total_d, xoff, yoff, tree, parent_pt, node_text):
    num_leafs = get_num_leaf_node(tree)
    depth = get_tree_depth(tree)
    root_text = list(tree.keys())[0]
    curr_pt = (xoff + (1 + num_leafs) / 2 / total_w, yoff)
    plot_mid_text(ax, curr_pt, parent_pt, node_text)
    plot_node(ax, root_text, curr_pt, parent_pt, decision_node)
    subdict = tree[root_text]
    yoff = yoff - 1 / total_d;
    for key in subdict.keys():
        xoff = xoff + 1 / total_w
        if type(subdict[key]) == str:
            plot_node(ax, subdict[key], (xoff, yoff), curr_pt, leaf_node)
            plot_mid_text(ax, (xoff, yoff), curr_pt, str(key))
        else:
            plot_tree(ax, total_w, total_d, xoff, yoff, subdict[key], curr_pt, str(key))

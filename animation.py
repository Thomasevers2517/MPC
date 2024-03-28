import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot_mechanism_3d(link_ends):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot links
    for i, link in enumerate(link_ends):
        ax.plot([link[0][0], link[1][0]],
                [link[0][1], link[1][1]],
                [link[0][2], link[1][2]], color='b')

    # Plot end points
    joints = np.unique(np.reshape(link_ends, (-1, 3)), axis=0)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

    # Set labels and title
    ax.set_xlabel('x_0')
    ax.set_ylabel('y_0')
    ax.set_zlabel('z_0')
    ax.set_title('SIP+FBM')

    plt.show()


def update(frame, link_ends_list, lines, points):
    link_ends_list_current = link_ends_list[frame]

    for i, link in enumerate(link_ends_list_current):
        # lines[i].set_data([link[0][0], link[1][0]],
        #                   [link[0][1], link[1][1]])
        # lines[i].set_3d_properties([link[0][2], link[1][2]])
        lines[i].set_data_3d([link[0][0], link[1][0]],
                             [link[0][1], link[1][1]],
                             [link[0][2], link[1][2]])

    joints = np.unique(np.reshape(link_ends_list_current, (-1, 3)), axis=0)
    points.set_offsets(np.c_[joints[:, 0], joints[:, 1]])
    points.set_3d_properties(joints[:, 2], 'z')

    return lines, points


def animate_mechanism_3d(link_ends_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    link_ends_start = link_ends_list[0]

    num_links = len(link_ends_start)

    lines = []

    # Plot links
    for i, link in enumerate(link_ends_start):
        line, = ax.plot([link[0][0], link[1][0]],
                        [link[0][1], link[1][1]],
                        [link[0][2], link[1][2]], color='b')
        lines.append(line)

    # Plot end points
    joints = np.unique(np.reshape(link_ends_start, (-1, 3)), axis=0)
    points = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

    # Set labels and title
    ax.set_xlabel('x_0')
    ax.set_ylabel('y_0')
    ax.set_zlabel('z_0')
    ax.set_title('SIP+FBM')
    ax.set_xlim(-0.1, 0.4)
    ax.set_ylim(0, 0.4)
    ax.set_zlim(-0.15, 0.15)
    ax.set_aspect('equal', adjustable='box')

    anim = FuncAnimation(fig, update, frames=len(link_ends_list),
                         fargs=(link_ends_list, lines, points), interval=1)

    plt.show()

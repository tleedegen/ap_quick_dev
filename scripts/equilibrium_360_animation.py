
import anchor_pro.main
import anchor_pro.plots as plts
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# Initialize controller
controller = anchor_pro.main.main()
item = controller.items_for_report['Base 02b']
matplotlib.use('TkAgg')

np.set_printoptions(2)


def plot_init(item, idx,init_idx,sf=1e6):
    theta = item.theta_z[idx]
    init = item.get_initial_dof_guess(theta)[init_idx]

    fig,w = plts._displaced_shape(item,init,theta)
    fig.show()


    fig, w = plts._equilibrium_plan_view(item,init, theta)
    fig.show()


def plot_solution(item, idx):
    theta = item.theta_z[idx]
    sol = item.equilibrium_solutions[:, idx]

    fig, w = plts._displaced_shape(item, sol, theta)
    fig.show()

idx = 0
init_idx = 0
plot_init(item, idx, init_idx)


# plot_solution(item, idx)
# fig, w = plts._equilibrium_plan_view(item,item.equilibrium_solutions[:,idx],item.theta_z[idx])

# idxs = [3]
#
# # for idx in idxs:
# #     plot_init(idx)
# #     plot_solution(idx)
#
# # fig, w = plts.base_anchors_vs_theta(item)
# # fig, W = plts.base_equilibrium(item)
# # fig, W = plts.base_displaced_shape(item)
# fig, w = plts.equipment_plan_view(item)

# Enable saving animation
# SAVE_ANIMATION = True  # Set to True to save as .mp4

# Get equilibrium solution shape
# num_steps = item.equilibrium_solutions.shape[1]

# Create figure and axis once
# fig, ax = plt.subplots()

# Placeholder for image
# img = None


# def update(i):
#     global img
#
#     u = item.equilibrium_solutions[:, i]
#     theta_z = item.theta_z[i]
#
#     # Generate new equilibrium plot (returns a new figure)
#     new_fig, _ = anchor_pro.plots._equilibrium_plan_view(item, u, theta_z)
#
#     # Extract image data from new figure
#     new_ax = new_fig.gca()
#     new_ax.set_frame_on(False)  # Hide frame
#
#     # Convert new figure to an array (avoids figure replacement)
#     fig.canvas.draw()
#     new_fig.canvas.draw()
#
#     # Grab pixel buffer from new figure
#     img_array = new_fig.canvas.renderer.buffer_rgba()
#
#     # Clear main figure and display new image
#     ax.clear()
#     img = ax.imshow(img_array)
#
#     plt.close(new_fig)  # Close the temporary figure


# Create animation
# ani = animation.FuncAnimation(fig, update, frames=num_steps, repeat=False, interval=500)

# Show animation
# plt.show()

# Save animation if enabled
# if SAVE_ANIMATION:
#     ani.save("equilibrium_animation.gif", writer="pillow", fps=10)
#     print("Animation saved as 'equilibrium_animation.mp4'.")

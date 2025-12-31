#!/usr/bin/env python3

# External
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class L2O_Visualization:
    def __init__(self):
        """Initialize with default styles and class variables for plot settings."""
        sns.set(style="whitegrid")  # Use Seaborn's whitegrid style by default
        self.independent_vars = []  # List of independent variables (x, z axes)
        self.dependent_vars = []    # List of dependent variables (y axis)
        self.fontsize = 10
        self.title_fontsize = 14
        self.num_cols = 2           # Number of plots horizontally (Two is nice for my display)

    def histogram_2D(self, data_dicts, variable, bins=30, titles=None, alpha=0.5, color_low="green", color_high="black"):
        """
        Visualize 2D histograms for a single variable with color based on values.

        Args:
            data_dicts (list of dict): List of dictionaries containing the data to plot.
            variable (str): The variable to plot.
            bins (int, optional): Number of bins for histograms. Defaults to 30.
            titles (list of str, optional): Titles for each plot. Defaults to None.
            alpha (float, optional): Transparency level for the histograms to overlay. Defaults to 0.5.
            color_low (str, optional): Color for the lower end of the colormap. Defaults to "green".
            color_high (str, optional): Color for the higher end of the colormap. Defaults to "black".
        """
        plot_count = len(data_dicts)

        # Calculate the number of rows needed (self.num_cols plots per row)
        n_rows = (plot_count + 1) // self.num_cols

        # Create subplots based on plot count and row/column setup
        fig, axes = plt.subplots(n_rows, self.num_cols, figsize=(8, 4 * n_rows))

        # Create a custom colormap
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [color_low, color_high])

        # Flatten axes array if it's multi-dimensional
        axes = axes.flatten()

        # Loop through each dictionary and plot histogram
        for i, data_dict in enumerate(data_dicts):
            title = titles[i] if titles and i < len(titles) else f'Plot {i + 1}'

            # Ensure the variable(s) exist in the data_dict
            if variable not in data_dict:
                print(f"Error: Variable '{variable}' not found in data for plot {i + 1}.")
                continue

            ax = axes[i]

            # Get the data for the variable
            data = np.asarray(data_dict[variable])

            # Create the histogram
            counts, bin_edges, _ = ax.hist(data, bins=bins, color='gray', alpha=alpha, label=variable)

            # Normalize the bin heights to map them to colors
            norm = Normalize(vmin=bin_edges.min(), vmax=bin_edges.max())
            bin_colors = custom_cmap(norm(bin_edges))

            # Color each bin with the corresponding color
            for j in range(len(counts)):
                ax.bar(bin_edges[j], counts[j], width=bin_edges[j + 1] - bin_edges[j], color=bin_colors[j], alpha=alpha)

            # Calculate and plot the mean line
            mean_value = np.mean(data)
            
            # Find the bin in which the mean value falls and use its color for the mean line
            bin_index = np.digitize(mean_value, bin_edges) - 1
            bin_index = np.clip(bin_index, 0, len(bin_colors) - 1)  # Ensure the index is within bounds
            mean_color = bin_colors[bin_index]

            ax.axvline(mean_value, color=mean_color, linestyle='-', label=f'{variable} Mean: {mean_value:.4f}')

            # Set labels and title
            ax.set_xlabel(variable, fontsize=self.fontsize)
            ax.set_ylabel('Frequency', fontsize=self.fontsize)
            ax.set_title(title, fontsize=14)
            ax.grid(True)

            # Add legend
            ax.legend()

        # Remove any unused subplots
        if plot_count < n_rows * self.num_cols:
            for j in range(plot_count, n_rows * self.num_cols):
                fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()


    def histogram_3D(self, data_dicts, variables, bins=10, titles=None, alpha=1.0, color_low="green", color_high="black"):
        """
        Visualize 3D histograms for variable pairs with color based on multivariable values.

        Args:
            data_dicts (list of dict): List of dictionaries containing the data to plot.
            variables (list of list of str): List of variable pairs (x, y) to plot.
            bins (int, optional): Number of bins for histograms. Defaults to 10.
            titles (list of str, optional): Titles for each plot. Defaults to None.
            alpha (float, optional): Transparency level for the histograms. Defaults to 0.7.
            color_low (str, optional): Color for the lower end of the colormap. Defaults to "green".
            color_high (str, optional): Color for the higher end of the colormap. Defaults to "black".
        """
        plot_count = len(data_dicts)

        # Calculate the number of rows needed (self.num_cols plots per row)
        n_rows = (plot_count + 1) // self.num_cols

        # Create a custom colormap
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [color_low, color_high])

        # Create subplots based on plot count and row/column setup
        fig = plt.figure(figsize=(13, 4.25 * n_rows))

        # Loop through each dictionary and plot histogram
        for i, data_dict in enumerate(data_dicts):
            title = titles[i] if titles and i < len(titles) else f'Plot {i+1}'

            # Ensure both variables exist in the data_dict
            if len(variables) != 2 or variables[0] not in data_dict or variables[1] not in data_dict:
                print(f"Error: Variables '{variables}' not found in data for plot {i + 1}.")
                continue

            # Extract data for the variables
            x_data = data_dict[variables[0]]
            y_data = data_dict[variables[1]]

            # Create 2D histogram for variable values with the given number of bins
            hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)

            # Create an X-Y mesh of the same dimension as the 2D data
            xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)

            # Flatten out the arrays for plotting
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros_like(xpos)

            # Flatten histogram data
            dz = hist.flatten()

            # Normalize the variable values (x_data + y_data) to map them to color intensities
            combined_values = (xpos + ypos).flatten()
            norm = Normalize(vmin=combined_values.min(), vmax=combined_values.max())
            colors = custom_cmap(norm(combined_values))  # Custom colormap based on color_low and color_high

            # Add subplot for each plot
            ax = fig.add_subplot(n_rows, self.num_cols, i + 1, projection='3d')

            # Set the bin width for both x and y axes
            dx = (xedges[1] - xedges[0]) * np.ones_like(zpos)
            dy = (yedges[1] - yedges[0]) * np.ones_like(zpos)

            # Plot the 3D bars with the color based on the combined variable values
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=alpha)

            # Set labels and title
            ax.set_xlabel(variables[0], fontsize=self.fontsize)
            ax.set_ylabel(variables[1], fontsize=self.fontsize)
            ax.set_zlabel('Frequency', fontsize=self.fontsize)
            ax.set_title(title, fontsize=self.title_fontsize)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

    def scatterplot_2D(self, data_dicts, variables, titles=None, xlabel=None, ylabel=None, alpha=0.7, color_low="green", color_high="black"):
        """
        Visualize 2D scatter plots comparing two variables with trendlines and mean indicators for the dependent variable at each unique x value.

        Args:
            data_dicts (list of dict): List of dictionaries containing the data to plot.
            variables (list of str): List containing the names of the two variables to plot.
            titles (list of str, optional): Titles for each subplot. Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to None.
            ylabel (str, optional): Label for the y-axis. Defaults to None.
            alpha (float, optional): Transparency level for the scatter plots. Defaults to 0.7.
            color_low (str, optional): Color for the lower end of the colormap. Defaults to "green".
            color_high (str, optional): Color for the higher end of the colormap. Defaults to "black".
        """
        plot_count = len(data_dicts)

        # Calculate the number of rows and columns needed (self.num_cols plots per row)
        n_rows = (plot_count + 1) // self.num_cols

        # Create a custom colormap if color_variable is provided
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [color_low, color_high])

        # Create subplots based on plot count and row/column setup
        fig, axes = plt.subplots(n_rows, self.num_cols, figsize=(7 * self.num_cols, 4 * n_rows))
        axes = axes.flatten()  # Flatten axes array if it's multi-dimensional

        # Loop through each dictionary and plot scatterplot
        for i, data_dict in enumerate(data_dicts):
            title = titles[i] if titles and i < len(titles) else f'Plot {i + 1}'

            # Ensure both variables exist in the data_dict
            if variables[0] not in data_dict or variables[1] not in data_dict:
                print(f"Error: Variables '{variables}' not found in data for plot {i + 1}.")
                continue

            # Extract the data for the two variables
            x_data = np.array(data_dict[variables[0]])  # Ensure this is a numpy array
            y_data = np.array(data_dict[variables[1]])  # Ensure this is a numpy array

            # Normalize the variable values (x_data + y_data + z_data) to map them to color intensities
            combined_values = (y_data).flatten()
            norm = Normalize(vmin=combined_values.min(), vmax=combined_values.max())
            colors = custom_cmap(norm(combined_values))

            # Create the scatter plot
            ax = axes[i]
            ax.scatter(x_data, y_data, alpha=alpha, c=colors, label='Data')

            # Calculate the mean of y_data for each unique x_data point
            unique_x = np.unique(x_data)
            mean_y_for_x = [np.mean(y_data[x_data == x]) for x in unique_x]

            # Plot the mean y value for each x data point
            ax.plot(unique_x, mean_y_for_x, 'ro-', label='Mean Y for each X')

            # Set the labels and title
            ax.set_xlabel(xlabel if xlabel else variables[0], fontsize=self.fontsize)
            ax.set_ylabel(ylabel if ylabel else variables[1], fontsize=self.fontsize)
            ax.set_title(title, fontsize=self.title_fontsize)

            # Add grid and legend
            ax.grid(True)
            ax.legend()

        # Remove any unused subplots
        if plot_count < n_rows * self.num_cols:
            for j in range(plot_count, n_rows * self.num_cols):
                fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def scatterplot_3d(self, data_dicts, variables, titles=None, xlabel=None, ylabel=None, zlabel=None, alpha=0.7, color_low="green", color_high="red"):
        """
        Visualize 3D scatter plots with optional colors based on a fourth variable.

        Args:
            data_dicts (list of dict): List of dictionaries containing the data to plot.
            variables (list of str): List containing the names of the three variables to plot (x, y, z).
            color_variable (str, optional): Name of the variable used to color the points. Defaults to None.
            titles (list of str, optional): Titles for each subplot. Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to None.
            ylabel (str, optional): Label for the y-axis. Defaults to None.
            zlabel (str, optional): Label for the z-axis. Defaults to None.
            alpha (float, optional): Transparency level for the scatter plots. Defaults to 0.7.
            color_low (str, optional): Color for the lower end of the colormap. Defaults to "green".
            color_high (str, optional): Color for the higher end of the colormap. Defaults to "red".
        """
        plot_count = len(data_dicts)

        # Calculate the number of rows and columns needed (self.num_cols plots per row)
        n_rows = (plot_count + 1) // self.num_cols

        # Create a custom colormap if color_variable is provided
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [color_low, color_high])

        # Create subplots based on plot count and row/column setup
        fig = plt.figure(figsize=(7 * self.num_cols, 6 * n_rows))

        # Loop through each dictionary and plot 3D scatterplot
        for i, data_dict in enumerate(data_dicts):
            title = titles[i] if titles and i < len(titles) else f'Plot {i + 1}'

            # Ensure all variables exist in the data_dict
            missing_vars = [var for var in variables if var not in data_dict]
            if missing_vars:
                print(f"Error: Missing variables {missing_vars} in data for plot {i + 1}.")
                continue

            # Extract the data for the three variables
            x_data = np.array(data_dict[variables[0]])  # X data
            y_data = np.array(data_dict[variables[1]])  # Y data
            z_data = np.array(data_dict[variables[2]])  # Z data

            # Normalize the variable values (x_data + y_data + z_data) to map them to color intensities
            combined_values = (x_data + y_data).flatten()
            norm = Normalize(vmin=combined_values.min(), vmax=combined_values.max())
            colors = custom_cmap(norm(combined_values))  # Custom colormap based on color_low and color_high

            # Add subplot for each plot in 3D
            ax = fig.add_subplot(n_rows, self.num_cols, i + 1, projection='3d')

            # Create the 3D scatter plot
            scatter = ax.scatter(x_data, y_data, z_data, alpha=alpha, c=colors)

            # Set the labels and title
            ax.set_xlabel(xlabel if xlabel else variables[0], fontsize=self.fontsize)
            ax.set_ylabel(ylabel if ylabel else variables[1], fontsize=self.fontsize)
            ax.set_zlabel(zlabel if zlabel else variables[2], fontsize=self.fontsize)
            ax.set_title(title, fontsize=self.title_fontsize)

            # Add grid
            ax.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()


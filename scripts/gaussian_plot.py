import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

c = np.linspace(0, 3, 100)  # Example range of c values
sigmas = [0.1, 0.5, 1.0]

for sigma in sigmas:
    gaussian = np.exp(-0.5 * c**2 / sigma**2)
    plt.plot(c, gaussian, label=f'sigma={sigma}')
    
plt.xlabel('c')
plt.ylabel('Gaussian value')
plt.legend()
plt.title('Gaussian Curve for Different Sigma Values')
plt.show()


# Example c values (replace with your actual c values)
c_values = np.random.normal(loc=1, scale=0.5, size=1000)  # Replace with your data

# Plot histogram
plt.hist(c_values, bins=30, alpha=0.7, label='Histogram', edgecolor='black', density=True)

# Add a density plot
sns.kdeplot(c_values, color='blue', label='Density')

# Add labels and title
plt.xlabel('c values')
plt.ylabel('Density')
plt.title('Distribution of c values')
plt.legend()
plt.show()

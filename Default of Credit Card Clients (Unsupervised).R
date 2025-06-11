# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Author: Soo Tong King, Pah Zhen Xiang
# Title: Credit Card Default Predictive Modelling
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Performance Metrics Function ----
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Load necessary libraries
library(readxl)       # For reading Excel files
library(ggplot2)      # For data visualization
library(dplyr)

# Load the dataset and skip the first row which is redundant
data <- read_excel("default of credit card clients.xls", skip = 1)

# Drop the 'ID' column which is not necessary for modelling
data <- data %>% select(-ID)

# Rename the PAY_0 column to PAY_1
data <- data %>%
  rename(PAY_1 = PAY_0)

# Inspect the structure and overview of the dataset
head(data)
str(data)

# Check for missing values 
sapply(data, function(x) sum(is.na(x)))

# Save a copy of the dataset for unsupervised learning
df <- as.data.frame(data)

# >>>>>>>>>>>>>>>>>>>>>>>>>>
# Unsupervised Learning ----
# >>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>
## Pre-Processing Data ----
# >>>>>>>>>>>>>>>>>>>>>>>>>

# Data preprocessing
# Remove the target variable from the dataset
unlabel_df <- subset(df, select = -c(`default payment next month`))

# Standardize the features to have mean = 0 and variance = 1
df_scaled <- scale(unlabel_df)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Principle Component Analysis (PCA) ----
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Perform PCA on the scaled data
pca_result <- prcomp(df_scaled)

# Summary of PCA results
summary(pca_result) # Provides the proportion of variance explained by each principal component

# Mean of each feature used in PCA
pca_result$center

# Check whether standardization was applied
pca_result$scale

# Eigenvectors (principal component directions) of the PCA
pca_result$rotation 

## Principal component scores (transformed data in the principal component space)
# pca_result$x

# Extract eigenvalues and create a data frame for plotting
eigenvalues <- pca_result$sdev^2
eigenvalues_df <- data.frame(PC = 1:length(eigenvalues), Eigenvalue = eigenvalues)

# Scree Plot to visualize eigenvalues of principal components
scree_plot <- ggplot(eigenvalues_df, aes(x = PC, y = Eigenvalue)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "steelblue", size = 2) +
  labs(title = "Scree Plot", x = "Principal Component", y = "Eigenvalue") +
  theme_minimal()
print(scree_plot)

# Calculate Proportion of Variance Explained (PVE) and Cumulative PVE
variance_explained <- eigenvalues / sum(eigenvalues)
cumulative_variance_explained <- cumsum(variance_explained)

pve_cpve_df <- data.frame(PC = 1:length(variance_explained), 
                          PVE = variance_explained, 
                          CPVE = cumulative_variance_explained)

# Plot PVE and Cumulative PVE (CPVE) vs Principal Components
pve_cpve_plot <- ggplot(pve_cpve_df, aes(x = PC)) +
  geom_line(aes(y = PVE, color = "PVE"), size = 1) +
  geom_point(aes(y = PVE, color = "PVE"), size = 2) +
  geom_line(aes(y = CPVE, color = "CPVE"), size = 1, linetype = "dashed") +
  geom_point(aes(y = CPVE, color = "CPVE"), size = 2) +
  scale_color_manual(name = "Metrics", values = c("PVE" = "blue", "CPVE" = "red")) +
  labs(title = "PVE and CPVE vs Principal Components", x = "Principal Components", y = "Variance Explained (%)") +
  theme_minimal()
print(pve_cpve_plot)

# Biplot of the first two principal components
# Suppress row names to avoid number labels in the plot
row.names(pca_result$x) <- rep(".", nrow(df_scaled))

biplot(pca_result, scale = 0)


# >>>>>>>>>>>>>>>>>>>>>>>
# k-Means Clustering ----
# >>>>>>>>>>>>>>>>>>>>>>>

# Function to calculate WSS for different numbers of clusters
wss <- function(k) {
  kmeans(df_scaled, centers = k, nstart = 10)$tot.withinss
}

# Calculate WSS for a range of cluster numbers (1 to 20)
k_values <- 1:20
wss_values <- sapply(k_values, wss)

# Plot WSS to determine the optimal number of clusters using the Elbow Method
ggplot(data.frame(K = k_values, WSS = wss_values), aes(x = K, y = WSS)) +
  geom_line() +
  geom_point() +
  ggtitle('Elbow Method for Optimal Number of Clusters') +
  labs(x = 'Number of Clusters (K)', y = 'Within-Cluster Sum of Squares (WSS)') +
  theme_minimal()

# Function to perform K-means clustering and add cluster assignments to the data
perform_kmeans <- function(k, data) {
  set.seed(42) # For reproducibility
  kmeans_result <- kmeans(data, centers = k, nstart = 25)
  data.frame(pca_result$x, Cluster = as.factor(kmeans_result$cluster), K = k)
}

# Perform PCA for dimensionality reduction
pca_result <- prcomp(df_scaled)
pca_df <- as.data.frame(pca_result$x)

# Loop through each value of K (1 to 5), perform clustering, and plot the results
for (k in 1:5) {
  # Perform K-means clustering
  cluster_data <- perform_kmeans(k, df_scaled)
  
  # Plot clusters for the current number of clusters
  p <- ggplot(cluster_data, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(size = 1, alpha = 0.7) +
    ggtitle(paste('K-means Clustering of Credit Card Clients for K =', k)) +
    labs(x = 'PC1', y = 'PC2') +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = 'white', color = 'white'),  
      plot.background = element_rect(fill = 'white', color = 'white'),   
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 8)
    )
  
  # Print the plot
  print(p)
}
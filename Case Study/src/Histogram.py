import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/University/Course/Grade4/CSDS340/Case Study/code/train.csv')

feature_names = df.columns[:-1] 
output_name = df.columns[-1]

# Plot histograms for each feature
for feature in feature_names:
    plt.figure(figsize=(10, 6))
    plt.hist(df[feature], bins=50, alpha=0.7, label=feature)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {feature}')
    plt.legend()
    plt.show()



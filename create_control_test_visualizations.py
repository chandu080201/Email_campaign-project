import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create directory for visualizations if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Create data for overall metrics
metrics = ['CTR', 'Open Rate', 'Click-to-Open Rate', 'Conversion Rate']
control_values = [17.92, 10.35, 20.48, 1.85]
test_values = [37.20, 15.72, 42.31, 4.12]
improvements = [(test - control) / control * 100 for control, test in zip(control_values, test_values)]

# 1. Overall Metrics Comparison
plt.figure(figsize=(12, 8))
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, control_values, width, label='Control Group', color='#3498db')
bars2 = ax.bar(x + width/2, test_values, width, label='Test Group', color='#2ecc71')

# Add labels and title
ax.set_ylabel('Percentage (%)', fontsize=14)
ax.set_title('Control vs. Test Group: Overall Metrics', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=12)

# Add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(bars1)
add_labels(bars2)

# Add improvement percentages
for i, improvement in enumerate(improvements):
    ax.annotate(f'+{improvement:.1f}%',
                xy=(i, max(control_values[i], test_values[i]) + 1),
                ha='center', va='bottom',
                color='red', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/control_vs_test_overall.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance by Purchase History
purchase_segments = ['0', '1-2', '3-5', '6-10', '10+']
control_by_purchase = [8.45, 12.76, 18.93, 25.41, 32.87]
test_by_purchase = [21.32, 28.54, 36.78, 48.92, 62.45]

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(purchase_segments))
width = 0.35

bars1 = ax.bar(x - width/2, control_by_purchase, width, label='Control Group', color='#3498db')
bars2 = ax.bar(x + width/2, test_by_purchase, width, label='Test Group', color='#2ecc71')

ax.set_ylabel('Click-Through Rate (%)', fontsize=14)
ax.set_xlabel('Number of Past Purchases', fontsize=14)
ax.set_title('Control vs. Test Group: CTR by Purchase History', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(purchase_segments, fontsize=12)
ax.legend(fontsize=12)

add_labels(bars1)
add_labels(bars2)

# Add improvement percentages
improvements = [(test - control) / control * 100 for control, test in zip(control_by_purchase, test_by_purchase)]
for i, improvement in enumerate(improvements):
    ax.annotate(f'+{improvement:.1f}%',
                xy=(i, max(control_by_purchase[i], test_by_purchase[i]) + 1),
                ha='center', va='bottom',
                color='red', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/control_vs_test_purchase_history.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Performance by Country
countries = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'ES']
control_by_country = [19.87, 18.92, 17.45, 16.89, 15.76, 14.32, 13.87]
test_by_country = [42.35, 40.12, 36.78, 35.42, 32.89, 29.87, 28.92]

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(countries))
width = 0.35

bars1 = ax.bar(x - width/2, control_by_country, width, label='Control Group', color='#3498db')
bars2 = ax.bar(x + width/2, test_by_country, width, label='Test Group', color='#2ecc71')

ax.set_ylabel('Click-Through Rate (%)', fontsize=14)
ax.set_xlabel('Country', fontsize=14)
ax.set_title('Control vs. Test Group: CTR by Country', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(countries, fontsize=12)
ax.legend(fontsize=12)

add_labels(bars1)
add_labels(bars2)

# Add improvement percentages
improvements = [(test - control) / control * 100 for control, test in zip(control_by_country, test_by_country)]
for i, improvement in enumerate(improvements):
    ax.annotate(f'+{improvement:.1f}%',
                xy=(i, max(control_by_country[i], test_by_country[i]) + 1),
                ha='center', va='bottom',
                color='red', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/control_vs_test_country.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Performance by Email Characteristics
# Email Version
versions = ['Personalized', 'Generic']
control_by_version = [21.45, 14.38]
test_by_version = [42.87, 28.76]

# Email Text
texts = ['Short Email', 'Long Email']
control_by_text = [19.87, 15.98]
test_by_text = [39.76, 32.45]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Email Version
x1 = np.arange(len(versions))
width = 0.35

bars1 = ax1.bar(x1 - width/2, control_by_version, width, label='Control Group', color='#3498db')
bars2 = ax1.bar(x1 + width/2, test_by_version, width, label='Test Group', color='#2ecc71')

ax1.set_ylabel('Click-Through Rate (%)', fontsize=14)
ax1.set_title('CTR by Email Version', fontsize=16)
ax1.set_xticks(x1)
ax1.set_xticklabels(versions, fontsize=12)
ax1.legend(fontsize=12)

for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Email Text
x2 = np.arange(len(texts))

bars3 = ax2.bar(x2 - width/2, control_by_text, width, label='Control Group', color='#3498db')
bars4 = ax2.bar(x2 + width/2, test_by_text, width, label='Test Group', color='#2ecc71')

ax2.set_ylabel('Click-Through Rate (%)', fontsize=14)
ax2.set_title('CTR by Email Text', fontsize=16)
ax2.set_xticks(x2)
ax2.set_xticklabels(texts, fontsize=12)
ax2.legend(fontsize=12)

for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/control_vs_test_email_characteristics.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Performance by Timing
# Day of Week
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
control_by_day = [16.78, 19.45, 21.32, 19.87, 17.54, 14.32, 13.87]
test_by_day = [35.42, 41.23, 44.87, 42.12, 36.78, 29.87, 28.92]

# Hour of Day
hours = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24']
control_by_hour = [12.34, 14.87, 21.45, 20.87, 18.32, 14.56]
test_by_hour = [25.67, 30.98, 45.32, 43.76, 38.45, 30.12]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Day of Week
x1 = np.arange(len(days))
width = 0.35

bars1 = ax1.bar(x1 - width/2, control_by_day, width, label='Control Group', color='#3498db')
bars2 = ax1.bar(x1 + width/2, test_by_day, width, label='Test Group', color='#2ecc71')

ax1.set_ylabel('Click-Through Rate (%)', fontsize=14)
ax1.set_title('CTR by Day of Week', fontsize=16)
ax1.set_xticks(x1)
ax1.set_xticklabels(days, fontsize=12)
ax1.legend(fontsize=12)

for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Hour of Day
x2 = np.arange(len(hours))

bars3 = ax2.bar(x2 - width/2, control_by_hour, width, label='Control Group', color='#3498db')
bars4 = ax2.bar(x2 + width/2, test_by_hour, width, label='Test Group', color='#2ecc71')

ax2.set_ylabel('Click-Through Rate (%)', fontsize=14)
ax2.set_xlabel('Hour of Day', fontsize=14)
ax2.set_title('CTR by Hour of Day', fontsize=16)
ax2.set_xticks(x2)
ax2.set_xticklabels(hours, fontsize=12)
ax2.legend(fontsize=12)

for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/control_vs_test_timing.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Interaction Effects: Email Version × Purchase History
purchase_segments = ['0', '1-2', '3-5', '6-10', '10+']

# Personalized
control_personalized = [9.87, 14.32, 21.45, 28.76, 36.54]
test_personalized = [23.45, 31.23, 40.87, 53.21, 67.89]

# Generic
control_generic = [7.12, 11.23, 16.54, 22.32, 29.45]
test_generic = [18.76, 25.43, 32.12, 43.76, 56.78]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Personalized
x1 = np.arange(len(purchase_segments))
width = 0.35

bars1 = ax1.bar(x1 - width/2, control_personalized, width, label='Control Group', color='#3498db')
bars2 = ax1.bar(x1 + width/2, test_personalized, width, label='Test Group', color='#2ecc71')

ax1.set_ylabel('Click-Through Rate (%)', fontsize=14)
ax1.set_title('CTR by Purchase History (Personalized Emails)', fontsize=16)
ax1.set_xticks(x1)
ax1.set_xticklabels(purchase_segments, fontsize=12)
ax1.legend(fontsize=12)

for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Generic
x2 = np.arange(len(purchase_segments))

bars3 = ax2.bar(x2 - width/2, control_generic, width, label='Control Group', color='#3498db')
bars4 = ax2.bar(x2 + width/2, test_generic, width, label='Test Group', color='#2ecc71')

ax2.set_ylabel('Click-Through Rate (%)', fontsize=14)
ax2.set_xlabel('Number of Past Purchases', fontsize=14)
ax2.set_title('CTR by Purchase History (Generic Emails)', fontsize=16)
ax2.set_xticks(x2)
ax2.set_xticklabels(purchase_segments, fontsize=12)
ax2.legend(fontsize=12)

for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/control_vs_test_interaction.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Feature Importance
features = ['Past Purchases', 'Email Version', 'Weekday', 'Hour', 'Country', 'Email Text', 'Version × Purchases', 'Weekday × Hour']
importance = [0.2876, 0.1932, 0.1245, 0.1187, 0.0987, 0.0876, 0.0654, 0.0243]

# Sort by importance
sorted_indices = np.argsort(importance)[::-1]
sorted_features = [features[i] for i in sorted_indices]
sorted_importance = [importance[i] for i in sorted_indices]

fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.barh(sorted_features, sorted_importance, color=sns.color_palette("viridis", len(features)))

ax.set_xlabel('Importance Score', fontsize=14)
ax.set_title('Feature Importance in the Optimization Model', fontsize=16)
ax.invert_yaxis()  # Highest importance at the top

# Add value labels
for i, v in enumerate(sorted_importance):
    ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations created successfully!")

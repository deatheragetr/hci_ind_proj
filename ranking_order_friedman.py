import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Data
data = {
    "id": "1761256540487",
    "text": "In a comma-delimited list, please rank the four interfaces above from favorite to least favorite. For example, if you preferred iStock the most, followed by B, C, and then A, you'd write: iStock, B, C, A",
    "answers": [
      "A,B,C,iStock",
      "A, iStock, C, B",
      "C,B,A,iStock",
      "C, B, A, iStock",
      "C, B, A, iStock",
      "C,A,iStock,B",
      "C, A, iStock, B",
      "A, iStock, B, C",
      "A,C,B, iStock",
      "A, C, B, iStock",
      "A, iStock, B, C",
      "A, iStock,  B, C",
      "iStock, A, B, C",
      "C,A,B,iStock",
      "B, A, C, iStock",
      "iStock,A,C,B",
      "A,B,C, iStock",
      "A, iStock, C, B",
      "A, iStock, B, C",
      "C, B, iStock, A",
      "C,A,B, iStock",
      "B,C,A, iStock",
      "C,A,iStock,B",
      "B, A, C, iStock",
      "C, B, iStock, A"
    ]
}

# Parse the rankings
def parse_ranking(answer_str):
    """Parse a comma-delimited ranking string into a list"""
    # Clean up the string and split by comma
    items = [item.strip() for item in answer_str.split(",")]
    
    # Handle the special case with typo "A B"
    if "AB" in items:
        # Assuming "AB" should be "A" followed by "B"
        idx = items.index("AB")
        items[idx:idx+1] = ["A", "B"]
    
    return items

# Process all rankings
rankings_list = []
for i, answer in enumerate(data["answers"]):
    ranking = parse_ranking(answer)
    for position, item in enumerate(ranking, 1):
        rankings_list.append({
            'respondent': i + 1,
            'interface': item,
            'rank': position
        })

df = pd.DataFrame(rankings_list)

# Create a matrix where rows are respondents and columns are interfaces
# Values are the ranks given by each respondent to each interface
rank_matrix = df.pivot_table(
    index='respondent', 
    columns='interface', 
    values='rank',
    fill_value=None
)

print("="*60)
print("RANK MATRIX (Respondent x Interface)")
print("="*60)
print(rank_matrix.head())
print()

# Calculate ranking counts for each interface at each position
rank_counts = df.pivot_table(
    index='interface', 
    columns='rank', 
    values='respondent',
    aggfunc='count',
    fill_value=0
)

print("="*60)
print("RANKING COUNTS")
print("(Number of times each interface was ranked in each position)")
print("="*60)
print(rank_counts)
print()

# Sort interfaces by their overall performance (most 1st place, then most 2nd place, etc.)
def sort_key(interface):
    # Return a tuple for hierarchical sorting: (-1st_place_count, -2nd_place_count, -3rd_place_count)
    return tuple(-rank_counts.loc[interface, i] if i in rank_counts.columns else 0 for i in range(1, 5))

sorted_interfaces = sorted(rank_counts.index, key=sort_key)
print("="*60)
print("INTERFACES ORDERED BY PERFORMANCE (best to worst):")
print("="*60)
for i, interface in enumerate(sorted_interfaces, 1):
    counts = [rank_counts.loc[interface, j] if j in rank_counts.columns else 0 for j in range(1, 5)]
    print(f"{i}. {interface}: 1st={counts[0]}, 2nd={counts[1]}, 3rd={counts[2]}, 4th={counts[3]}")
print()

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

print("="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

# 1. FRIEDMAN TEST (Omnibus test for differences among interfaces)
print("\n1. FRIEDMAN TEST (Omnibus Test)")
print("-" * 40)

# Prepare data for Friedman test - need the rank matrix
# The Friedman test expects data where rows are blocks (respondents) and columns are treatments (interfaces)
interfaces = ['A', 'B', 'C', 'iStock']
friedman_data = rank_matrix[interfaces].values

# Perform Friedman test
statistic, p_value = stats.friedmanchisquare(*[friedman_data[:, i] for i in range(len(interfaces))])

print(f"Friedman Test Statistic (Chi-square): {statistic:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Degrees of freedom: {len(interfaces) - 1}")

alpha = 0.05
if p_value < alpha:
    print(f"\nResult: SIGNIFICANT (p < {alpha})")
    print("There are statistically significant differences among the interfaces.")
    perform_posthoc = True
else:
    print(f"\nResult: NOT SIGNIFICANT (p >= {alpha})")
    print("No statistically significant differences among the interfaces.")
    perform_posthoc = False

# Calculate mean ranks for interpretation
mean_ranks = {}
for interface in interfaces:
    mean_ranks[interface] = rank_matrix[interface].mean()

print("\nMean Ranks (lower is better):")
for interface in sorted(mean_ranks.keys(), key=lambda x: mean_ranks[x]):
    print(f"  {interface}: {mean_ranks[interface]:.2f}")

# 2. POST-HOC ANALYSIS: Pairwise Wilcoxon Signed-Rank Tests
if perform_posthoc:
    print("\n" + "="*60)
    print("2. POST-HOC PAIRWISE COMPARISONS")
    print("(Wilcoxon Signed-Rank Tests with Bonferroni Correction)")
    print("-" * 40)
    
    # Get all pairs of interfaces
    pairs = list(combinations(interfaces, 2))
    n_comparisons = len(pairs)
    
    # Bonferroni correction
    bonferroni_alpha = alpha / n_comparisons
    print(f"\nNumber of comparisons: {n_comparisons}")
    print(f"Original alpha level: {alpha}")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
    print()
    
    # Store results for summary
    pairwise_results = []
    
    for interface1, interface2 in pairs:
        ranks1 = rank_matrix[interface1].values
        ranks2 = rank_matrix[interface2].values
        
        # Perform Wilcoxon signed-rank test
        # Use 'wilcox' method for exact p-values when possible
        try:
            statistic, p_val = stats.wilcoxon(ranks1, ranks2, method='exact')
        except:
            # Fall back to normal approximation if exact method fails
            statistic, p_val = stats.wilcoxon(ranks1, ranks2)
        
        # Calculate effect size (r = Z / sqrt(N))
        # For Wilcoxon test, we can approximate Z from the test statistic
        n = len(ranks1)
        z_score = (statistic - (n * (n + 1) / 4)) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        effect_size = abs(z_score) / np.sqrt(n)
        
        # Determine significance
        is_significant = p_val < bonferroni_alpha
        sig_symbol = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < bonferroni_alpha else ""
        
        pairwise_results.append({
            'Pair': f"{interface1} vs {interface2}",
            'Mean Rank 1': mean_ranks[interface1],
            'Mean Rank 2': mean_ranks[interface2],
            'Better': interface1 if mean_ranks[interface1] < mean_ranks[interface2] else interface2,
            'Statistic': statistic,
            'p-value': p_val,
            'Significant': is_significant,
            'Symbol': sig_symbol,
            'Effect Size (r)': effect_size
        })
        
        print(f"{interface1} vs {interface2}:")
        print(f"  Mean ranks: {interface1}={mean_ranks[interface1]:.2f}, {interface2}={mean_ranks[interface2]:.2f}")
        print(f"  Better interface: {pairwise_results[-1]['Better']}")
        print(f"  Wilcoxon statistic: {statistic:.2f}")
        print(f"  p-value: {p_val:.6f} {sig_symbol}")
        print(f"  Effect size (r): {effect_size:.3f}")
        print(f"  Significant at Bonferroni-corrected level: {'YES' if is_significant else 'NO'}")
        print()
    
    # Create summary table
    print("\n" + "="*60)
    print("SUMMARY OF PAIRWISE COMPARISONS")
    print("-" * 40)
    
    results_df = pd.DataFrame(pairwise_results)
    results_df = results_df.sort_values('p-value')
    
    print("\nPairwise Comparison Results (sorted by p-value):")
    print(results_df[['Pair', 'Better', 'p-value', 'Symbol', 'Effect Size (r)']].to_string(index=False))
    
    print("\n" + "-" * 40)
    print("Significance levels:")
    print("  *** p < 0.001")
    print("  **  p < 0.01")
    print(f"  *   p < {bonferroni_alpha:.4f} (Bonferroni-corrected)")
    
    print("\nEffect size interpretation (Cohen's guidelines):")
    print("  Small:  r = 0.10")
    print("  Medium: r = 0.30")
    print("  Large:  r = 0.50")
    
    # Final ranking based on statistical results
    print("\n" + "="*60)
    print("FINAL STATISTICAL RANKING")
    print("-" * 40)
    
    # Count significant wins for each interface
    wins = {interface: 0 for interface in interfaces}
    for result in pairwise_results:
        if result['Significant']:
            wins[result['Better']] += 1
    
    print("\nNumber of significant pairwise wins:")
    for interface in sorted(wins.keys(), key=lambda x: (-wins[x], mean_ranks[x])):
        print(f"  {interface}: {wins[interface]} wins (mean rank: {mean_ranks[interface]:.2f})")

else:
    print("\n" + "="*60)
    print("POST-HOC TESTS NOT PERFORMED")
    print("(No significant omnibus effect detected)")

# ============================================================
# VISUALIZATION (Original charts with modifications)
# ============================================================

# Create the bump chart
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for each interface
colors = {
    'A': '#2E86AB',      # Blue
    'B': '#A23B72',      # Purple
    'C': '#F18F01',      # Orange
    'iStock': '#C73E1D'  # Red
}

# Set up positions for interfaces on y-axis (0 = top, 3 = bottom)
y_positions = {interface: i for i, interface in enumerate(sorted_interfaces)}

# Plot lines for each interface showing their count at each rank position
for interface in sorted_interfaces:
    x_values = []  # Rank positions (1, 2, 3, 4)
    y_values = []  # Count at each rank
    
    for rank in range(1, 5):
        x_values.append(rank)
        # Get the count for this interface at this rank
        count = rank_counts.loc[interface, rank] if rank in rank_counts.columns else 0
        y_values.append(count)
    
    # Plot the line with markers
    ax.plot(x_values, y_values, 
           color=colors[interface], 
           linewidth=2.5,
           marker='o',
           markersize=10,
           label=interface,
           alpha=0.8)
    
    # Add value labels at each point
    for x, y in zip(x_values, y_values):
        ax.annotate(f'{int(y)}', 
                   (x, y), 
                   textcoords="offset points", 
                   xytext=(0, 10),
                   ha='center',
                   fontsize=10,
                   color=colors[interface],
                   fontweight='bold')

# Formatting
ax.set_xlabel('Rank Position', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Responses', fontsize=14, fontweight='bold')

# Add statistical significance to title if applicable
if perform_posthoc:
    title = f'Interface Ranking Bump Chart\n(Friedman test p={p_value:.4f}, significant differences detected)'
else:
    title = f'Interface Ranking Bump Chart\n(Friedman test p={p_value:.4f}, no significant differences)'
ax.set_title(title, fontsize=14, fontweight='bold')

# Set x-axis
ax.set_xlim(0.5, 4.5)
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['1st\n(Best)', '2nd', '3rd', '4th\n(Worst)'], fontsize=12)

# Set y-axis
ax.set_ylim(-1, max(rank_counts.max()) + 2)
ax.set_ylabel('Number of Responses', fontsize=14, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3, linestyle=':', axis='y')
ax.set_axisbelow(True)

# Add legend with interface ordering and mean ranks
legend_labels = [f"{interface} (Mean rank: {mean_ranks[interface]:.2f})" 
                 for interface in sorted_interfaces]
ax.legend(loc='upper left', title='Interface (ordered by performance)', 
         framealpha=0.95, fontsize=11)

plt.tight_layout()
plt.savefig('./rank_order_bump_chart_with_stats.png', dpi=300, bbox_inches='tight')
plt.show()

# Create an alternative version with interfaces on y-axis
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Plot lines connecting each interface's position across ranks
for i, interface in enumerate(sorted_interfaces):
    x_values = []  # Rank positions
    y_values = []  # Y position (interface position)
    sizes = []     # Size of marker based on count
    
    for rank in range(1, 5):
        x_values.append(rank)
        y_values.append(i)
        # Get the count for sizing the marker
        count = rank_counts.loc[interface, rank] if rank in rank_counts.columns else 0
        sizes.append(count * 50)  # Scale for visibility
    
    # Plot the line
    ax2.plot(x_values, y_values, 
            color=colors[interface], 
            linewidth=2,
            alpha=0.6,
            zorder=1)
    
    # Plot markers with size based on count
    for x, y, size, rank in zip(x_values, y_values, sizes, range(1, 5)):
        count = rank_counts.loc[interface, rank] if rank in rank_counts.columns else 0
        ax2.scatter(x, y, s=size, 
                   color=colors[interface], 
                   alpha=0.8,
                   zorder=2)
        # Add count labels
        if count > 0:
            ax2.annotate(f'{int(count)}', 
                       (x, y), 
                       ha='center',
                       va='center',
                       fontsize=10,
                       color='white',
                       fontweight='bold',
                       zorder=3)

# Formatting
ax2.set_xlabel('Rank Position', fontsize=14, fontweight='bold')
ax2.set_ylabel('Interface', fontsize=14, fontweight='bold')
ax2.set_title('Interface Ranking Bump Chart - Alternative View\n(Y-axis shows interfaces ordered by performance, bubble size shows count)', 
             fontsize=16, fontweight='bold')

# Set x-axis
ax2.set_xlim(0.5, 4.5)
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['1st\n(Best)', '2nd', '3rd', '4th\n(Worst)'], fontsize=12)

# Set y-axis with interface names
ax2.set_ylim(-0.5, len(sorted_interfaces) - 0.5)
ax2.set_yticks(range(len(sorted_interfaces)))
ax2.set_yticklabels(sorted_interfaces, fontsize=12)
ax2.invert_yaxis()  # Best at top

# Add grid
ax2.grid(True, alpha=0.3, linestyle=':', axis='x')
ax2.set_axisbelow(True)

# Add note about bubble sizes
ax2.text(0.98, 0.02, 'Bubble size = number of responses',
        transform=ax2.transAxes,
        fontsize=10,
        ha='right',
        style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('./rank_order_bump_chart_alt_with_stats.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n" + "="*60)
print(f"Total respondents: {len(data['answers'])}")
print("\nCharts saved as:")
print("- rank_order_bump_chart_with_stats.png (count vs rank position)")
print("- rank_order_bump_chart_alt_with_stats.png (interfaces on y-axis with bubble sizes)")
print("="*60)

# Save statistical results to CSV
if perform_posthoc and 'results_df' in locals():
    results_df.to_csv('pairwise_comparison_results.csv', index=False)
    print("\nStatistical results saved to: pairwise_comparison_results.csv")

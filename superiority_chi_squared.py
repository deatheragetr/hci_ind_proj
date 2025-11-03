#!/usr/bin/env python3
"""
Likert Scale Data Analysis and Visualization
Comparing four interfaces: iStock, Prototype A, B, and C
Modified to include Chi-squared test for overall significance
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data
data_json = '''
[  {
    "id": "1761255796072",
    "text": "iStock: This interface is probably superior to any other existing interface I've used (e.g., Google Images, Dall-e) for finding or generating a useful image",
    "answers": [
      "3",
      "3",
      "1",
      "2",
      "2",
      "5",
      "4",
      "3",
      "4",
      "3",
      "3",
      "4",
      "3",
      "3",
      "2",
      "4",
      "3",
      "3",
      "2",
      "3",
      "3",
      "4",
      "3",
      "2",
      "1"
    ]
  },
  {
    "id": "1761253456036",
    "text": "Prototype A: This interface would probably be superior to any existing interface I've used (e.g., iStock, Google Images, Dall-e) for finding or generating a useful image",
    "answers": [
      "3",
      "3",
      "5",
      "3",
      "3",
      "4",
      "4",
      "3",
      "4",
      "4",
      "4",
      "3",
      "4",
      "4",
      "2",
      "3",
      "4",
      "4",
      "4",
      "3",
      "4",
      "4",
      "4",
      "5",
      "3"
    ]
  },
  {
    "id": "1761256143399",
    "text": "Prototype B: This interface would probably be superior to any existing interface I've used (e.g., iStock, Google Images, Dall-e) for finding or generating a useful image",
    "answers": [
      "2",
      "2",
      "5",
      "5",
      "4",
      "4",
      "2",
      "5",
      "4",
      "4",
      "3",
      "3",
      "3",
      "3",
      "5",
      "4",
      "3",
      "4",
      "2",
      "4",
      "4",
      "4",
      "2",
      "5",
      "5"
    ]
  },
  {
    "id": "1761256422360",
    "text": "Prototype C: This interface would probably be superior to any existing interface I've used (e.g., iStock, Google Images, Dall-e) for finding or generating a useful image",
    "answers": [
      "4",
      "4",
      "5",
      "5",
      "4",
      "4",
      "5",
      "2",
      "4",
      "3",
      "2",
      "2",
      "3",
      "4",
      "5",
      "4",
      "3",
      "4",
      "2",
      "4",
      "5",
      "3",
      "4",
      "4",
      "5"
    ]
  }
]
'''

def load_and_prepare_data():
    """Load JSON data and prepare it for analysis"""
    data = json.loads(data_json)
    
    # Extract interface names and convert answers to integers
    interfaces = []
    all_answers = []
    
    for item in data:
        # Extract interface name from the text
        if "iStock:" in item["text"]:
            name = "iStock"
        elif "Prototype A:" in item["text"]:
            name = "Prototype A"
        elif "Prototype B:" in item["text"]:
            name = "Prototype B"
        elif "Prototype C:" in item["text"]:
            name = "Prototype C"
        
        interfaces.append(name)
        all_answers.append([int(x) for x in item["answers"]])
    
    # Create DataFrame
    df_list = []
    for interface, answers in zip(interfaces, all_answers):
        for i, answer in enumerate(answers):
            df_list.append({
                'Interface': interface,
                'Respondent': i + 1,
                'Score': answer
            })
    
    df = pd.DataFrame(df_list)
    return df, interfaces, all_answers

def calculate_statistics(df, interfaces, all_answers):
    """Calculate statistical measures for each interface"""
    stats_data = []
    
    for interface, answers in zip(interfaces, all_answers):
        answers_array = np.array(answers)
        stats_data.append({
            'Interface': interface,
            'Mean': np.mean(answers_array),
            'Median': np.median(answers_array),
            'Mode': stats.mode(answers_array, keepdims=False).mode,
            'Std Dev': np.std(answers_array, ddof=1),
            'Min': np.min(answers_array),
            'Max': np.max(answers_array),
            'Q1': np.percentile(answers_array, 25),
            'Q3': np.percentile(answers_array, 75),
            'IQR': np.percentile(answers_array, 75) - np.percentile(answers_array, 25),
            'Sample Size': len(answers_array)
        })
    
    stats_df = pd.DataFrame(stats_data)
    return stats_df

def create_visualizations(df, stats_df):
    """Create multiple visualizations for Likert scale data"""
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Box Plot - Great for comparing distributions
    ax1 = plt.subplot(2, 3, 1)
    df.boxplot(column='Score', by='Interface', ax=ax1)
    ax1.set_title('Distribution Comparison - Box Plot')
    ax1.set_xlabel('Interface')
    ax1.set_ylabel('Likert Score (1-5)')
    ax1.set_ylim(0.5, 5.5)
    plt.sca(ax1)
    plt.xticks(rotation=45)
    
    # 2. Violin Plot - Shows distribution shape
    ax2 = plt.subplot(2, 3, 2)
    interfaces_order = ['iStock', 'Prototype A', 'Prototype B', 'Prototype C']
    sns.violinplot(data=df, x='Interface', y='Score', order=interfaces_order, ax=ax2)
    ax2.set_title('Distribution Shape - Violin Plot')
    ax2.set_xlabel('Interface')
    ax2.set_ylabel('Likert Score (1-5)')
    ax2.set_ylim(0.5, 5.5)
    plt.sca(ax2)
    plt.xticks(rotation=45)
    
    # 3. Bar Chart with Error Bars (Mean ± Std Dev)
    ax3 = plt.subplot(2, 3, 3)
    x_pos = np.arange(len(stats_df))
    ax3.bar(x_pos, stats_df['Mean'], yerr=stats_df['Std Dev'], 
            capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax3.set_xlabel('Interface')
    ax3.set_ylabel('Mean Score')
    ax3.set_title('Intelligibility: Mean Scores with Standard Deviation')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(stats_df['Interface'], rotation=45)
    ax3.set_ylim(0, 5.5)
    ax3.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral (3)')
    ax3.legend()
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(stats_df['Mean'], stats_df['Std Dev'])):
        ax3.text(i, mean + std + 0.1, f'{mean:.2f}', ha='center', va='bottom')
    
    # 4. Stacked Bar Chart - Response Distribution
    ax4 = plt.subplot(2, 3, 4)
    response_counts = {}
    for interface in interfaces_order:
        interface_df = df[df['Interface'] == interface]
        counts = interface_df['Score'].value_counts().sort_index()
        response_counts[interface] = [counts.get(i, 0) for i in range(1, 6)]
    
    response_df = pd.DataFrame(response_counts, index=['1 Strongly Disagree', '2 Disagree', '3 Neutral', '4 Agree', '5 Strongly Agree'])
    response_df_pct = response_df.div(response_df.sum(axis=0), axis=1) * 100
    
    response_df_pct.T.plot(kind='bar', stacked=True, ax=ax4, 
                           color=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#1f77b4'])
    ax4.set_xlabel('Interface')
    ax4.set_ylabel('Percentage of Responses (%)')
    ax4.set_title('Response Distribution (Stacked %)')
    ax4.legend(title='Score', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.sca(ax4)
    plt.xticks(rotation=45)
    
    # 5. Diverging Bar Chart (Centered at Neutral)
    ax5 = plt.subplot(2, 3, 5)
    
    # Calculate percentages for negative (1-2), neutral (3), and positive (4-5)
    diverging_data = []
    for interface in interfaces_order:
        interface_df = df[df['Interface'] == interface]
        total = len(interface_df)
        negative = len(interface_df[interface_df['Score'] <= 2]) / total * 100
        neutral = len(interface_df[interface_df['Score'] == 3]) / total * 100
        positive = len(interface_df[interface_df['Score'] >= 4]) / total * 100
        diverging_data.append({
            'Interface': interface,
            'Negative': -negative,
            'Neutral': neutral,
            'Positive': positive
        })
    
    div_df = pd.DataFrame(diverging_data)
    y_pos = np.arange(len(div_df))
    
    # Plot diverging bars
    ax5.barh(y_pos, div_df['Negative'], color='#d62728', label='Disagree (1-2)')
    ax5.barh(y_pos, div_df['Positive'], color='#2ca02c', label='Agree (4-5)')
    ax5.barh(y_pos, div_df['Neutral'], left=div_df['Negative'], color='#ffbb78', label='Neutral (3)')
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(div_df['Interface'])
    ax5.set_xlabel('Percentage (%)')
    ax5.set_title('Agreement Analysis (Diverging)')
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax5.legend()
    ax5.set_xlim(-60, 80)
    
    # 6. Heatmap of Mean Scores
    ax6 = plt.subplot(2, 3, 6)
    
    # Create a comparison matrix
    comparison_matrix = stats_df.set_index('Interface')[['Mean', 'Median', 'Mode', 'Std Dev']]
    sns.heatmap(comparison_matrix.T, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=3, vmin=1, vmax=5, ax=ax6, cbar_kws={'label': 'Score'})
    ax6.set_title('Statistical Measures Heatmap')
    ax6.set_xlabel('Interface')
    ax6.set_ylabel('Measure')
    
    plt.tight_layout()
    plt.savefig('likert_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_statistical_tests(all_answers, interfaces):
    """Perform statistical tests to compare interfaces"""
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    # 1. Chi-squared Test of Independence
    print("\n1. Chi-squared Test of Independence:")
    print("-" * 40)
    
    # Create contingency table for chi-squared test
    # Rows: Interfaces, Columns: Likert scores (1-5)
    contingency_table = []
    for answers in all_answers:
        row = [answers.count(i) for i in range(1, 6)]
        contingency_table.append(row)
    
    contingency_array = np.array(contingency_table)
    
    # Display contingency table first (before test)
    print("\n  Contingency Table (Observed Frequencies):")
    print("  " + "-" * 50)
    print(f"  {'Interface':<15} {'1':<6} {'2':<6} {'3':<6} {'4':<6} {'5':<6} {'Total':<6}")
    print("  " + "-" * 50)
    for i, interface in enumerate(interfaces):
        row_str = f"  {interface:<15}"
        for j in range(5):
            row_str += f" {contingency_array[i, j]:<6}"
        row_str += f" {contingency_array[i].sum():<6}"
        print(row_str)
    print("  " + "-" * 50)
    totals_str = "  Totals:        "
    for j in range(5):
        totals_str += f" {contingency_array[:, j].sum():<6}"
    totals_str += f" {contingency_array.sum():<6}"
    print(totals_str)
    
    # Check if we can perform chi-squared test
    # Remove columns (scores) that have all zeros or very low frequencies
    non_zero_cols = []
    col_labels = []
    for j in range(5):
        col_sum = contingency_array[:, j].sum()
        if col_sum > 0:  # Only include columns with at least one response
            non_zero_cols.append(j)
            col_labels.append(str(j + 1))
    
    if len(non_zero_cols) < 2:
        print("\n  Warning: Insufficient variation in scores to perform chi-squared test.")
        print("  All interfaces have very similar response patterns.")
        chi2_p_value = 1.0  # No significance if test can't be performed
    else:
        # Create reduced contingency table with only non-zero columns
        reduced_contingency = contingency_array[:, non_zero_cols]
        
        try:
            # Perform chi-squared test on reduced table
            chi2_stat, chi2_p_value, dof, expected_freq = stats.chi2_contingency(reduced_contingency)
            
            print(f"\n  Chi-squared statistic: {chi2_stat:.4f}")
            print(f"  Degrees of freedom: {dof}")
            print(f"  P-value: {chi2_p_value:.4f}")
            
            if chi2_p_value < 0.05:
                print("  Result: Significant association between interface and ratings (p < 0.05)")
                print("         The distribution of responses differs across interfaces.")
            else:
                print("  Result: No significant association (p >= 0.05)")
                print("         Response distributions are similar across interfaces.")
            
            # Check if expected frequencies are adequate
            min_expected = expected_freq.min()
            if min_expected < 5:
                print(f"\n  Warning: Minimum expected frequency is {min_expected:.2f} (< 5).")
                print("  Chi-squared test may not be reliable. Consider Fisher's exact test.")
            
            # Calculate and display Cramér's V for effect size
            n = reduced_contingency.sum()
            min_dim = min(reduced_contingency.shape[0] - 1, reduced_contingency.shape[1] - 1)
            if min_dim > 0:
                cramers_v = np.sqrt(chi2_stat / (n * min_dim))
                print(f"\n  Cramér's V (effect size): {cramers_v:.4f}")
                if cramers_v < 0.1:
                    effect = "negligible"
                elif cramers_v < 0.3:
                    effect = "small"
                elif cramers_v < 0.5:
                    effect = "medium"
                else:
                    effect = "large"
                print(f"  Effect size interpretation: {effect}")
                
        except ValueError as e:
            print(f"\n  Warning: Could not perform chi-squared test: {str(e)}")
            print("  This typically occurs when expected frequencies are too low.")
            print("  Consider using Fisher's exact test or combining categories.")
            chi2_p_value = 1.0  # No significance if test can't be performed
    
    # 2. Pairwise Kolmogorov-Smirnov Tests (only if chi-squared is significant)
    if chi2_p_value < 0.05:
        print("\n2. Pairwise Kolmogorov-Smirnov Tests:")
        print("-" * 40)
        print("  (Performed because chi-squared test showed overall significance)")
        print()
        
        # Perform pairwise K-S comparisons
        for i in range(len(interfaces)):
            for j in range(i+1, len(interfaces)):
                ks_stat, p_val = stats.ks_2samp(all_answers[i], all_answers[j])
                print(f"  {interfaces[i]} vs {interfaces[j]}:")
                print(f"    K-S statistic: {ks_stat:.4f}, p-value: {p_val:.4f}", end="")
                if p_val < 0.05:
                    print(" *")
                else:
                    print()
    else:
        print("\n2. Pairwise Kolmogorov-Smirnov Tests:")
        print("-" * 40)
        print("  (Skipped: chi-squared test showed no overall significance)")

def main():
    """Main function to run all analyses"""
    print("="*60)
    print("LIKERT SCALE DATA ANALYSIS")
    print("Comparing Interface Superiority Ratings")
    print("="*60)
    
    # Load and prepare data
    df, interfaces, all_answers = load_and_prepare_data()
    
    # Calculate statistics
    stats_df = calculate_statistics(df, interfaces, all_answers)
    
    # Display statistics
    print("\nDESCRIPTIVE STATISTICS:")
    print("-"*60)
    print(stats_df.to_string(index=False))
    
    # Perform statistical tests
    perform_statistical_tests(all_answers, interfaces)
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)
    create_visualizations(df, stats_df)
    
    # Summary insights
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    # Find best performing interface
    best_interface = stats_df.loc[stats_df['Mean'].idxmax(), 'Interface']
    best_mean = stats_df.loc[stats_df['Mean'].idxmax(), 'Mean']
    
    print(f"\n1. Highest rated interface: {best_interface} (Mean = {best_mean:.2f})")
    
    # Rank interfaces by mean score
    ranked = stats_df.sort_values('Mean', ascending=False)
    print("\n2. Ranking by mean score:")
    for idx, row in ranked.iterrows():
        print(f"   {idx+1}. {row['Interface']}: {row['Mean']:.2f} (SD = {row['Std Dev']:.2f})")
    
    # Check for agreement (low standard deviation)
    most_agreed = stats_df.loc[stats_df['Std Dev'].idxmin(), 'Interface']
    lowest_std = stats_df.loc[stats_df['Std Dev'].idxmin(), 'Std Dev']
    print(f"\n3. Most consensus: {most_agreed} (SD = {lowest_std:.2f})")
    
    print("\n" + "="*60)
    print("Analysis complete! Check 'likert_analysis.png' for visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Survey Data Analysis and Visualization Script
Analyzes coded qualitative survey data and creates comprehensive visualizations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


FILE_PATH = './likes.json'
# FILE_PATH = './dislikes.json'

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filepath):
    """Load and parse the survey data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_treatment_name(text):
    """Extract treatment name from the question text"""
    if "iStock" in text:
        return "iStock (Baseline)"
    elif "Prototype A" in text:
        return "Prototype A"
    elif "Prototype B" in text:
        return "Prototype B"
    elif "Prototype C" in text:
        return "Prototype C"
    return "Unknown"

def prepare_dataframes(data):
    """Convert raw data into structured dataframes for analysis"""
    
    # Lists to store processed data
    granular_data = []
    thematic_data = []
    
    for treatment in data:
        treatment_name = extract_treatment_name(treatment['text'])
        
        # Process each response
        for i, (themes, codes) in enumerate(zip(treatment['coded_answer_groups'], 
                                                treatment['coded_answers'])):
            # Granular codes
            for code in codes:
                granular_data.append({
                    'treatment': treatment_name,
                    'response_id': f"{treatment['id']}_{i}",
                    'code': code
                })
            
            # Thematic groups
            for theme in themes:
                thematic_data.append({
                    'treatment': treatment_name,
                    'response_id': f"{treatment['id']}_{i}",
                    'theme': theme
                })
    
    df_granular = pd.DataFrame(granular_data)
    df_thematic = pd.DataFrame(thematic_data)
    
    return df_granular, df_thematic

# def create_heatmap_visualization(df_granular, df_thematic):
#     """Create heatmap showing code frequency by treatment"""
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
#
#     # Granular codes heatmap
#     if not df_granular.empty:
#         granular_pivot = pd.crosstab(df_granular['code'], df_granular['treatment'])
#
#         # Sort by total frequency
#         granular_pivot['total'] = granular_pivot.sum(axis=1)
#         granular_pivot = granular_pivot.sort_values('total', ascending=True).drop('total', axis=1)
#
#         # Create heatmap with annotations
#         # sns.heatmap(granular_pivot, annot=True, fmt='d', cmap='Reds', 
#         #             ax=ax1, cbar_kws={'label': 'Frequency'})
#         # ax1.set_title('Granular Dislikes by Treatment', fontsize=14, fontweight='bold')
#         ax1.set_xlabel('Treatment', fontsize=12)
#         ax1.set_ylabel('Code', fontsize=12)
#
#     # Thematic groups heatmap
#     if not df_thematic.empty:
#         thematic_pivot = pd.crosstab(df_thematic['theme'], df_thematic['treatment'])
#
#         # Sort by total frequency
#         thematic_pivot['total'] = thematic_pivot.sum(axis=1)
#         thematic_pivot = thematic_pivot.sort_values('total', ascending=True).drop('total', axis=1)
#
#         sns.heatmap(thematic_pivot, annot=True, fmt='d', cmap='OrRd', 
#                     ax=ax2, cbar_kws={'label': 'Frequency'})
#         ax2.set_title('Thematic Dislikes by Treatment', fontsize=14, fontweight='bold')
#         ax2.set_xlabel('Treatment', fontsize=12)
#         ax2.yaxis.set_label_position("right")
#         ax2.yaxis.tick_right()
#         ax2.set_ylabel('Theme', fontsize=12)
#
#     # plt.suptitle('Dislikes Frequency Heatmaps', fontsize=16, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     return fig


def create_heatmap_visualization(df_granular, df_thematic):
    """Create heatmap showing code frequency by treatment (shared cbar in middle)."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1, 0.03, 1], wspace=0.05
    )  # middle col for colorbar

    ax1 = fig.add_subplot(gs[0, 0])   # left: granular
    cax = fig.add_subplot(gs[0, 1])   # middle: colorbar
    ax2 = fig.add_subplot(gs[0, 2])   # right: thematic

    # ---- Granular (LIMITED TO TOP 27)
    if not df_granular.empty:
        # Get top 27 codes by frequency
        top_27_codes = df_granular['code'].value_counts().head(27).index
        df_granular_top27 = df_granular[df_granular['code'].isin(top_27_codes)]
        
        g = pd.crosstab(df_granular_top27['code'], df_granular_top27['treatment'])
        g = g.assign(total=g.sum(1)).sort_values('total').drop(columns='total')
    else:
        g = pd.DataFrame()

    # ---- Thematic
    if not df_thematic.empty:
        t = pd.crosstab(df_thematic['theme'], df_thematic['treatment'])
        t = t.sort_index()  # Sort alphabetically by theme name
    else:
        t = pd.DataFrame()

    # Shared color scale (only if both exist; otherwise use that one's range)
    if not g.empty and not t.empty:
        vmin = min(g.to_numpy().min(), t.to_numpy().min())
        vmax = max(g.to_numpy().max(), t.to_numpy().max())
    elif not g.empty:
        vmin, vmax = g.to_numpy().min(), g.to_numpy().max()
    else:
        vmin, vmax = t.to_numpy().min(), t.to_numpy().max()

    # Left heatmap (no colorbar)
    if not g.empty:
        sns.heatmap(
            g, annot=True, fmt='d', cmap='Reds', ax=ax1, cbar=False,
            vmin=vmin, vmax=vmax
        )
        # ax1.set_xlabel('Treatment', fontsize=12)
        # ax1.set_ylabel('Code', fontsize=12)
        ax1.tick_params(axis='y', labelrotation=0)

    # Right heatmap; its colorbar goes in the middle axis
    if not t.empty:
        hm = sns.heatmap(
            t, annot=True, fmt='d', cmap='OrRd', ax=ax2,
            cbar_ax=cax, #cbar_kws={'label': 'Frequency'},
            vmin=vmin, vmax=vmax
        )
        # ax2.set_title('Thematic Dislikes by Treatment', fontsize=14, fontweight='bold')
        # ax2.set_xlabel('Treatment', fontsize=12)

        # Put y-axis and its label on the right side, nicely centered
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        # ax2.set_ylabel('Theme', fontsize=12, rotation=-90, va='center', labelpad=20)
        ax2.tick_params(axis='y', labelrotation=0)

        # Optional: smaller ticks on the colorbar
        cax.tick_params(labelsize=10)

    plt.tight_layout()
    return fig

def create_stacked_bar_chart(df_granular, df_thematic, data):
    """Create stacked bar charts showing proportion of codes by treatment"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Get top codes for better visualization
    top_codes = df_granular['code'].value_counts().head(15).index if not df_granular.empty else []
    df_granular_top = df_granular[df_granular['code'].isin(top_codes)]
    
    # Granular codes stacked bar
    if not df_granular_top.empty:
        granular_crosstab = pd.crosstab(df_granular_top['treatment'], 
                                        df_granular_top['code'], normalize='index') * 100
        
        granular_crosstab.plot(kind='bar', stacked=True, ax=ax1, 
                               colormap='tab20', width=0.7)
        ax1.set_title('Distribution of Top 15 Dislikes by Treatment (%)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Treatment', fontsize=12)
        ax1.set_ylabel('Percentage of Responses', fontsize=12)
        ax1.legend(title='Code', bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=9, title_fontsize=10)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Thematic groups stacked bar
    if not df_thematic.empty:
        thematic_crosstab = pd.crosstab(df_thematic['treatment'], 
                                       df_thematic['theme'], normalize='index') * 100
        
        thematic_crosstab.plot(kind='bar', stacked=True, ax=ax2, 
                              colormap='Set3', width=0.7)
        ax2.set_title('Distribution of Thematic Dislikes by Treatment (%)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Treatment', fontsize=12)
        ax2.set_ylabel('Percentage of Responses', fontsize=12)
        ax2.legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=9, title_fontsize=10)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Dislikes Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_network_graph(data):
    """Create network graph showing relationships between themes and codes"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    treatments = []
    for treatment_data in data:
        treatment_name = extract_treatment_name(treatment_data['text'])
        treatments.append((treatment_name, treatment_data))
    
    for idx, (treatment_name, treatment_data) in enumerate(treatments):
        ax = axes[idx]
        
        # Create graph
        G = nx.Graph()
        
        # Add treatment node (central)
        G.add_node(treatment_name, node_type='treatment', size=1000)
        
        # Track connections
        theme_counts = defaultdict(int)
        code_counts = defaultdict(int)
        theme_code_connections = defaultdict(int)
        
        # Process all responses
        for themes, codes in zip(treatment_data['coded_answer_groups'], 
                               treatment_data['coded_answers']):
            for theme in themes:
                theme_counts[theme] += 1
                G.add_node(theme, node_type='theme')
                G.add_edge(treatment_name, theme)
                
                for code in codes:
                    code_counts[code] += 1
                    G.add_node(code, node_type='code')
                    G.add_edge(theme, code)
                    theme_code_connections[(theme, code)] += 1
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes by type with different colors and sizes
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            if G.nodes[node].get('node_type') == 'treatment':
                node_colors.append('#DC2626')  # Red for treatment
                node_sizes.append(2000)
            elif G.nodes[node].get('node_type') == 'theme':
                node_colors.append('#EA580C')  # Orange for themes
                node_sizes.append(800 + theme_counts.get(node, 0) * 20)
            else:  # code
                node_colors.append('#FCA5A5')  # Light red for codes
                node_sizes.append(300 + code_counts.get(node, 0) * 10)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7, ax=ax)
        
        # Draw edges with varying thickness based on frequency
        edges = G.edges()
        edge_weights = []
        for edge in edges:
            if edge[0] == treatment_name or edge[1] == treatment_name:
                edge_weights.append(2)
            else:
                weight = theme_code_connections.get(edge, 
                        theme_code_connections.get((edge[1], edge[0]), 1))
                edge_weights.append(min(weight * 0.5, 5))
        
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, ax=ax)
        
        # Add labels for themes and treatment only (codes would be too cluttered)
        labels = {}
        for node in G.nodes():
            if G.nodes[node].get('node_type') in ['treatment', 'theme']:
                # Truncate long labels
                label = node[:20] + '...' if len(node) > 20 else node
                labels[node] = label
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                               font_weight='bold', ax=ax)
        
        ax.set_title(f'{treatment_name}', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add legend for first subplot only
        if idx == 0:
            legend_elements = [
                plt.scatter([], [], c='#DC2626', s=200, alpha=0.7, label='Treatment'),
                plt.scatter([], [], c='#EA580C', s=150, alpha=0.7, label='Theme'),
                plt.scatter([], [], c='#FCA5A5', s=100, alpha=0.7, label='Dislike Code')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.suptitle('Network Graph: Treatment-Theme-Dislike Relationships', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_comparison_matrix(df_granular, df_thematic):
    """Create a comparison matrix showing top differences between treatments"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Calculate top distinctive codes for each treatment
    if not df_granular.empty:
        # Get proportion of each code by treatment
        code_props = pd.crosstab(df_granular['code'], df_granular['treatment'], 
                                 normalize='columns')
        
        # Find codes with highest variance across treatments
        code_props['variance'] = code_props.var(axis=1)
        top_variable_codes = code_props.nlargest(12, 'variance').drop('variance', axis=1)
        
        # Create heatmap
        sns.heatmap(top_variable_codes, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0.15, ax=ax1, cbar_kws={'label': 'Proportion'})
        ax1.set_title('Most Distinctive Dislikes\n(Highest Variance Across Treatments)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Treatment', fontsize=11)
        ax1.set_ylabel('Code', fontsize=11)
    
    # Calculate distinctive themes
    if not df_thematic.empty:
        theme_props = pd.crosstab(df_thematic['theme'], df_thematic['treatment'], 
                                 normalize='columns')
        
        # Create heatmap for all themes (usually fewer than codes)
        sns.heatmap(theme_props, annot=True, fmt='.2f', cmap='PuBuGn', 
                   ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Theme Distribution\n(Proportion by Treatment)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Treatment', fontsize=11)
        ax2.set_ylabel('Theme', fontsize=11)
    
    plt.suptitle('Treatment Comparison Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def generate_summary_statistics(df_granular, df_thematic, data):
    """Generate and print summary statistics"""
    
    print("\n" + "="*80)
    print("SURVEY DISLIKES DATA ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print("\n📊 OVERALL STATISTICS:")
    print(f"  • Total number of responses: {sum(len(t['coded_answers']) for t in data)}")
    print(f"  • Number of treatments: {len(data)}")
    print(f"  • Total granular code instances: {len(df_granular)}")
    print(f"  • Total thematic group instances: {len(df_thematic)}")
    print(f"  • Unique granular codes: {df_granular['code'].nunique() if not df_granular.empty else 0}")
    print(f"  • Unique thematic groups: {df_thematic['theme'].nunique() if not df_thematic.empty else 0}")
    
    # Per treatment statistics
    print("\n📈 PER TREATMENT STATISTICS:")
    for treatment in data:
        treatment_name = extract_treatment_name(treatment['text'])
        # Use coded_answers length since answers may be empty
        n_responses = len(treatment['coded_answers'])
        
        # Count codes
        total_codes = sum(len(codes) for codes in treatment['coded_answers'])
        total_themes = sum(len(themes) for themes in treatment['coded_answer_groups'])
        
        print(f"\n  {treatment_name}:")
        print(f"    • Responses: {n_responses}")
        print(f"    • Total code assignments: {total_codes}")
        print(f"    • Total theme assignments: {total_themes}")
        if n_responses > 0:
            print(f"    • Avg codes per response: {total_codes/n_responses:.2f}")
            print(f"    • Avg themes per response: {total_themes/n_responses:.2f}")
        else:
            print(f"    • Avg codes per response: N/A (no response text)")
            print(f"    • Avg themes per response: N/A (no response text)")
    
    # Top codes overall
    if not df_granular.empty:
        print("\n🚫 TOP 10 MOST FREQUENT DISLIKES OVERALL:")
        top_codes = df_granular['code'].value_counts().head(10)
        for i, (code, count) in enumerate(top_codes.items(), 1):
            print(f"    {i}. {code}: {count} occurrences")
    
    # Top themes overall
    if not df_thematic.empty:
        print("\n🎯 MOST FREQUENT THEMES OVERALL:")
        top_themes = df_thematic['theme'].value_counts()
        for i, (theme, count) in enumerate(top_themes.items(), 1):
            print(f"    {i}. {theme}: {count} occurrences")
    
    print("\n" + "="*80)

def main():
    """Main execution function"""
    
    # Load data
    print("Loading survey dislikes data...")
    filepath = FILE_PATH
    data = load_data(filepath)
    
    # Prepare dataframes
    print("Processing data...")
    df_granular, df_thematic = prepare_dataframes(data)
    
    # Generate summary statistics
    generate_summary_statistics(df_granular, df_thematic, data)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Heatmap visualization
    print("  • Creating heatmap visualization...")
    fig1 = create_heatmap_visualization(df_granular, df_thematic)
    fig1.savefig('./dislikes_heatmap_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Stacked bar chart
    print("  • Creating stacked bar charts...")
    fig2 = create_stacked_bar_chart(df_granular, df_thematic, data)
    fig2.savefig('./dislikes_stacked_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Network graph
    print("  • Creating network graph...")
    fig3 = create_network_graph(data)
    fig3.savefig('./dislikes_network_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Comparison matrix
    print("  • Creating comparison matrix...")
    fig4 = create_comparison_matrix(df_granular, df_thematic)
    fig4.savefig('./dislikes_comparison_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # # Export processed data to CSV for further analysis
    # print("  • Exporting processed data to CSV...")
    # df_granular.to_csv('/mnt/user-data/outputs/dislikes_granular_codes.csv', index=False)
    # df_thematic.to_csv('/mnt/user-data/outputs/dislikes_thematic_groups.csv', index=False)
    #
    # # Create a summary report
    # print("  • Creating summary report...")
    # with open('/mnt/user-data/outputs/dislikes_analysis_summary.txt', 'w') as f:
    #     f.write("SURVEY DISLIKES DATA ANALYSIS SUMMARY\n")
    #     f.write("="*80 + "\n\n")
    #
    #     # Write overall statistics
    #     f.write("OVERALL STATISTICS:\n")
    #     f.write(f"• Total responses: {sum(len(t['coded_answers']) for t in data)}\n")
    #     f.write(f"• Number of treatments: {len(data)}\n")
    #     f.write(f"• Unique granular codes: {df_granular['code'].nunique() if not df_granular.empty else 0}\n")
    #     f.write(f"• Unique themes: {df_thematic['theme'].nunique() if not df_thematic.empty else 0}\n\n")
    #
    #     # Write per-treatment breakdown
    #     f.write("PER TREATMENT BREAKDOWN:\n")
    #     for treatment in data:
    #         treatment_name = extract_treatment_name(treatment['text'])
    #         n_responses = len(treatment['coded_answers'])
    #         total_codes = sum(len(codes) for codes in treatment['coded_answers'])
    #         total_themes = sum(len(themes) for themes in treatment['coded_answer_groups'])
    #
    #         f.write(f"\n{treatment_name}:\n")
    #         f.write(f"  • Responses: {n_responses}\n")
    #         f.write(f"  • Code assignments: {total_codes}\n")
    #         f.write(f"  • Theme assignments: {total_themes}\n")
    #         if n_responses > 0:
    #             f.write(f"  • Avg codes/response: {total_codes/n_responses:.2f}\n")
    #             f.write(f"  • Avg themes/response: {total_themes/n_responses:.2f}\n")
    #         else:
    #             f.write(f"  • Avg codes/response: N/A\n")
    #             f.write(f"  • Avg themes/response: N/A\n")
    #
    # print("\n✅ Dislikes analysis complete! Files saved to /mnt/user-data/outputs/")
    # print("\nGenerated files:")
    # print("  📊 dislikes_heatmap_visualization.png - Code frequency heatmaps")
    # print("  📊 dislikes_stacked_bar_chart.png - Distribution analysis")
    # print("  📊 dislikes_network_graph.png - Relationship networks")
    # print("  📊 dislikes_comparison_matrix.png - Treatment comparisons")
    # print("  📄 dislikes_granular_codes.csv - Processed granular code data")
    # print("  📄 dislikes_thematic_groups.csv - Processed thematic group data")
    # print("  📄 dislikes_analysis_summary.txt - Statistical summary report")

if __name__ == "__main__":
    main()

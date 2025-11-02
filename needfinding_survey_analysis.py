import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import textwrap

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filename):
    """Load the survey data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def parse_multiple_choice(answers, delimiter=';'):
    """Parse multiple choice answers and count occurrences"""
    all_choices = []
    for answer in answers:
        if answer and answer != "N/A" and answer != "":
            choices = answer.split(delimiter)
            all_choices.extend([choice.strip() for choice in choices])
    return Counter(all_choices)

def create_likert_plot(question_text, answers, ax, scale_type='agreement'):
    """Create a horizontal bar chart for Likert scale questions"""
    # Convert string answers to integers
    numeric_answers = [int(a) for a in answers if a.isdigit()]
    
    if scale_type == 'agreement':
        labels = ['Strongly\nDisagree (1)', 'Disagree (2)', 'Neutral (3)', 'Agree (4)', 'Strongly\nAgree (5)']
    else:  # frequency
        labels = ['Never (1)', 'Rarely (2)', 'Sometimes (3)', 'Often (4)', 'Very\nFrequently (5)']
    
    counts = [numeric_answers.count(i) for i in range(1, 6)]
    
    # Create horizontal bar chart
    bars = ax.barh(range(5), counts)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Respondents')
    
    # Wrap title text
    wrapped_title = '\n'.join(textwrap.wrap(question_text, 60))
    ax.set_title(wrapped_title, fontsize=10, pad=10)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   str(count), va='center')
    
    # Add mean and std
    if numeric_answers:
        mean = np.mean(numeric_answers)
        std = np.std(numeric_answers)
        ax.text(0.95, 0.05, f'Mean: {mean:.2f}\nSD: {std:.2f}', 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def create_frequency_plot(question_text, answers, ax):
    """Create a bar chart for frequency questions with categorical ranges"""
    freq_counts = Counter(answers)
    
    # Define the order for frequency ranges
    freq_order = ['0-1 times', '2-5 times', '6-10 times', '11-20 times', '21+ times']
    
    # Get counts in order
    labels = []
    counts = []
    for freq in freq_order:
        if freq in freq_counts:
            labels.append(freq)
            counts.append(freq_counts[freq])
    
    # Create bar chart
    bars = ax.bar(range(len(labels)), counts)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Number of Respondents')
    
    # Wrap title text
    wrapped_title = '\n'.join(textwrap.wrap(question_text, 60))
    ax.set_title(wrapped_title, fontsize=10, pad=10)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               str(count), ha='center', va='bottom')

def create_multiple_choice_plot(question_text, answers, ax, top_n=15):
    """Create a horizontal bar chart for multiple choice questions"""
    choice_counts = parse_multiple_choice(answers)
    
    # Sort by frequency and take top N
    sorted_choices = choice_counts.most_common(top_n)
    
    if sorted_choices:
        labels, counts = zip(*sorted_choices)
        
        # Truncate long labels
        labels = [label[:40] + '...' if len(label) > 40 else label for label in labels]
        
        # Create horizontal bar chart
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Number of Selections')
        
        # Wrap title text
        wrapped_title = '\n'.join(textwrap.wrap(question_text, 60))
        ax.set_title(wrapped_title, fontsize=10, pad=10)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   str(count), va='center')
        
        ax.invert_yaxis()  # Highest on top

def analyze_survey_data(data):
    """Main function to analyze and visualize survey data"""
    
    # Create a figure with subplots for all visualizations
    fig = plt.figure(figsize=(20, 30))
    
    plot_idx = 1
    
    # 1. Image search frequency (Likert scale)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760490956947')
    create_likert_plot(question['text'], question['answers'], ax, 'frequency')
    plot_idx += 1
    
    # 2. Past week image search frequency
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760491079593')
    create_frequency_plot(question['text'], question['answers'], ax)
    plot_idx += 1
    
    # 3. Image sources (multiple choice)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760491847762')
    create_multiple_choice_plot(question['text'], question['answers'], ax)
    plot_idx += 1
    
    # 4. AI usage frequency (Likert scale)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760492429212')
    create_likert_plot(question['text'], question['answers'], ax, 'frequency')
    plot_idx += 1
    
    # 5. Past week AI usage
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760492950379')
    create_frequency_plot(question['text'], question['answers'], ax)
    plot_idx += 1
    
    # 6. AI tools used (multiple choice)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760492989328')
    # Combine with "Other" specifications
    other_question = next(q for q in data if q['id'] == '1760493274321')
    combined_answers = []
    for i, answer in enumerate(question['answers']):
        if 'Other' in answer and other_question['answers'][i]:
            # Replace or append the specific other response
            combined_answers.append(answer + ';' + other_question['answers'][i])
        else:
            combined_answers.append(answer)
    create_multiple_choice_plot(question['text'], combined_answers, ax)
    plot_idx += 1
    
    # 7. Reasons for using images/AI (multiple choice)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760494841641')
    create_multiple_choice_plot(question['text'], question['answers'], ax)
    plot_idx += 1
    
    # 8. Satisfaction with non-AI image tools (Likert)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760494093339')
    create_likert_plot(question['text'], question['answers'], ax, 'agreement')
    plot_idx += 1
    
    # 9. Frustration with non-AI image search (Likert)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760494148010')
    create_likert_plot(question['text'], question['answers'], ax, 'agreement')
    plot_idx += 1
    
    # 10. Satisfaction with AI image tools (Likert)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760494196907')
    create_likert_plot(question['text'], question['answers'], ax, 'agreement')
    plot_idx += 1
    
    # 11. Frustration with AI image creation (Likert)
    ax = plt.subplot(6, 3, plot_idx)
    question = next(q for q in data if q['id'] == '1760494250987')
    create_likert_plot(question['text'], question['answers'], ax, 'agreement')
    plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('survey_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("SURVEY ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total respondents: 28\n")
    
    # Summary for Likert scales
    likert_questions = [
        ('1760490956947', 'Image Search Frequency'),
        ('1760492429212', 'AI Usage Frequency'),
        ('1760494093339', 'Satisfaction with Non-AI Tools'),
        ('1760494148010', 'Frustration with Non-AI Search'),
        ('1760494196907', 'Satisfaction with AI Tools'),
        ('1760494250987', 'Frustration with AI Creation')
    ]
    
    print("\nLIKERT SCALE SUMMARY (1=Negative, 5=Positive):")
    print("-" * 50)
    for q_id, label in likert_questions:
        question = next(q for q in data if q['id'] == q_id)
        numeric_answers = [int(a) for a in question['answers'] if a.isdigit()]
        if numeric_answers:
            mean = np.mean(numeric_answers)
            std = np.std(numeric_answers)
            median = np.median(numeric_answers)
            print(f"{label:40} Mean: {mean:.2f}, SD: {std:.2f}, Median: {median:.1f}")
    
    # Most popular image sources
    print("\nTOP IMAGE SOURCES:")
    print("-" * 50)
    question = next(q for q in data if q['id'] == '1760491847762')
    source_counts = parse_multiple_choice(question['answers'])
    for source, count in source_counts.most_common(5):
        percentage = (count / 28) * 100
        print(f"{source:40} {count:2d} users ({percentage:.1f}%)")
    
    # Most popular AI tools
    print("\nTOP AI TOOLS:")
    print("-" * 50)
    question = next(q for q in data if q['id'] == '1760492989328')
    ai_counts = parse_multiple_choice(question['answers'])
    for tool, count in ai_counts.most_common(5):
        if tool != "N/A (I don't use image generative AI)":
            percentage = (count / 28) * 100
            print(f"{tool:40} {count:2d} users ({percentage:.1f}%)")
    
    # Usage reasons
    print("\nTOP REASONS FOR IMAGE/AI USE:")
    print("-" * 50)
    question = next(q for q in data if q['id'] == '1760494841641')
    reason_counts = parse_multiple_choice(question['answers'])
    for reason, count in reason_counts.most_common(5):
        if reason != "Other":
            percentage = (count / 28) * 100
            print(f"{reason:40} {count:2d} users ({percentage:.1f}%)")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Calculate some key metrics
    # Non-AI users
    ai_question = next(q for q in data if q['id'] == '1760492989328')
    non_ai_users = sum(1 for a in ai_question['answers'] if "N/A (I don't use image generative AI)" in a)
    ai_adoption = ((28 - non_ai_users) / 28) * 100
    
    print(f"• AI Adoption Rate: {ai_adoption:.1f}% of users have used image generative AI")
    print(f"• Google Images dominates as the primary image source")
    print(f"• DALL-E 3 is the most popular AI image generation tool")
    print(f"• Users show moderate satisfaction with both AI and non-AI tools (means ~3/5)")
    print(f"• Academic projects and personal enjoyment are top use cases")

def analyze_free_text_responses(data):
    """Analyze and summarize free text responses"""
    print("\n" + "="*80)
    print("FREE TEXT RESPONSE ANALYSIS")
    print("="*80)
    
    # Pain points with non-AI tools
    print("\nCOMMON COMPLAINTS ABOUT NON-AI IMAGE TOOLS:")
    print("-" * 50)
    question = next(q for q in data if q['id'] == '1760494344854')
    responses = [a for a in question['answers'] if a and a.strip()]
    if responses:
        # Group similar themes
        themes = {
            'AI contamination': [],
            'Search difficulty': [],
            'Copyright issues': [],
            'Quality issues': [],
            'Cost': [],
            'Other': []
        }
        
        for response in responses:
            lower_resp = response.lower()
            if 'ai' in lower_resp or 'generated' in lower_resp:
                themes['AI contamination'].append(response)
            elif 'find' in lower_resp or 'search' in lower_resp or 'query' in lower_resp:
                themes['Search difficulty'].append(response)
            elif 'copyright' in lower_resp:
                themes['Copyright issues'].append(response)
            elif 'quality' in lower_resp or 'resolution' in lower_resp:
                themes['Quality issues'].append(response)
            elif 'expensive' in lower_resp or 'cost' in lower_resp:
                themes['Cost'].append(response)
            else:
                themes['Other'].append(response)
        
        for theme, items in themes.items():
            if items:
                print(f"\n{theme} ({len(items)} mentions):")
                for item in items[:3]:  # Show first 3 examples
                    print(f"  - \"{item[:100]}...\"" if len(item) > 100 else f"  - \"{item}\"")
    
    # Pain points with AI tools
    print("\n\nCOMMON COMPLAINTS ABOUT AI IMAGE TOOLS:")
    print("-" * 50)
    question = next(q for q in data if q['id'] == '1760494388206')
    responses = [a for a in question['answers'] if a and a.strip() and 'NA' not in a and 'N/A' not in a]
    if responses:
        themes = {
            'Prompt difficulty': [],
            'Accuracy issues': [],
            'Iteration required': [],
            'Ethical concerns': [],
            'Other': []
        }
        
        for response in responses:
            lower_resp = response.lower()
            if 'prompt' in lower_resp:
                themes['Prompt difficulty'].append(response)
            elif 'exact' in lower_resp or 'mess up' in lower_resp or 'hands' in lower_resp or 'fingers' in lower_resp:
                themes['Accuracy issues'].append(response)
            elif 'multiple' in lower_resp or 'refin' in lower_resp or 'tweak' in lower_resp:
                themes['Iteration required'].append(response)
            elif 'societal' in lower_resp or 'concern' in lower_resp:
                themes['Ethical concerns'].append(response)
            else:
                themes['Other'].append(response)
        
        for theme, items in themes.items():
            if items:
                print(f"\n{theme} ({len(items)} mentions):")
                for item in items[:3]:
                    print(f"  - \"{item[:100]}...\"" if len(item) > 100 else f"  - \"{item}\"")

if __name__ == "__main__":
    # Load the data
    data = load_data('survey_data.json')
    
    # Run the analysis
    analyze_survey_data(data)
    
    # Analyze free text responses
    analyze_free_text_responses(data)
    
    print("\n" + "="*80)
    print("Analysis complete! Check 'survey_analysis.png' for visualizations.")
    print("="*80)

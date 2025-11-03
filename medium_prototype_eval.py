import json
import numpy as np
import pandas as pd
from collections import Counter
import statistics

def load_survey_data(filename):
    """Load survey data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def parse_likert_responses(answers):
    """Convert Likert scale responses to integers, handling empty strings"""
    numeric_answers = []
    for ans in answers:
        if ans and ans.isdigit():
            numeric_answers.append(int(ans))
    return numeric_answers

def calculate_likert_stats(answers, question_text):
    """Calculate statistics for Likert scale questions"""
    numeric_answers = parse_likert_responses(answers)
    
    if not numeric_answers:
        return None
    
    stats = {
        'question': question_text,
        'n': len(numeric_answers),
        'mean': round(np.mean(numeric_answers), 2),
        'median': np.median(numeric_answers),
        'mode': statistics.mode(numeric_answers) if numeric_answers else None,
        'std_dev': round(np.std(numeric_answers), 2),
        'min': min(numeric_answers),
        'max': max(numeric_answers),
        'frequency_distribution': dict(Counter(numeric_answers))
    }
    
    # Calculate percentage for each response option
    total = len(numeric_answers)
    percentages = {}
    for i in range(1, 6):
        count = numeric_answers.count(i)
        percentages[i] = f"{count} ({round(count/total*100, 1)}%)"
    stats['response_percentages'] = percentages
    
    return stats

def analyze_categorical_question(answers, question_text):
    """Analyze categorical/frequency questions"""
    # Filter out empty answers
    valid_answers = [ans for ans in answers if ans]
    
    if not valid_answers:
        return None
    
    freq_dist = Counter(valid_answers)
    total = len(valid_answers)
    
    stats = {
        'question': question_text,
        'n': total,
        'unique_responses': len(freq_dist),
        'frequency_distribution': {}
    }
    
    # Sort by frequency
    for answer, count in freq_dist.most_common():
        percentage = round(count/total*100, 1)
        stats['frequency_distribution'][answer] = f"{count} ({percentage}%)"
    
    return stats

def print_likert_stats(stats):
    """Pretty print Likert scale statistics"""
    print("\n" + "="*80)
    print(f"Question: {stats['question'][:70]}...")
    print("-"*80)
    print(f"N = {stats['n']} responses")
    print(f"Mean: {stats['mean']} | Median: {stats['median']} | Mode: {stats['mode']}")
    print(f"Std Dev: {stats['std_dev']} | Range: {stats['min']}-{stats['max']}")
    print("\nResponse Distribution:")
    print("  1 (Strongly Disagree) -> 5 (Strongly Agree)")
    for i in range(1, 6):
        bar_length = int(stats['frequency_distribution'].get(i, 0) * 2)
        bar = "█" * bar_length
        print(f"  {i}: {stats['response_percentages'][i]:12} {bar}")

def print_categorical_stats(stats):
    """Pretty print categorical statistics"""
    print("\n" + "="*80)
    print(f"Question: {stats['question'][:70]}...")
    print("-"*80)
    print(f"N = {stats['n']} responses")
    print(f"Unique response options: {stats['unique_responses']}")
    print("\nFrequency Distribution:")
    for answer, count_pct in stats['frequency_distribution'].items():
        print(f"  {answer}: {count_pct}")

def main():
    # Load the data
    data = load_survey_data('./medium_fidelity_survey_data.json')
    
    # Define which questions are Likert scale (based on inspection of the data)
    likert_questions = {
        '1761785374980': 'I found this interface easy to understand.',
        '1761785233587': 'I found this interface confusing.',
        '1761785779577': 'I found this interface allowed me to confidently explore and play around with different possibilities.',
        '1761786275667': 'I found myself not understanding what I was supposed to do next',
        '1761786525744': 'I feel this interface would save me time on my projects',
        '1761786603944': 'I enjoyed using this interface.',
        '1761788899675': 'This interface is superior to the existing tools I\'ve used',
        '1761786702008': 'I would use this interface if it were available.',
        '1761792482425': 'I am now more confident this interface would improve my current workflows',
        '1761844232183': 'How often do you search for images online (1=Never to 5=Very Often)'
    }
    
    # Define categorical questions
    categorical_questions = {
        '1761844268687': 'In the past week, how many times have you searched for an image?',
        '1761787078568': 'Understanding improvement after animated prototype',
        '1761786494728': 'How well does this interface address your needs?'
    }
    
    # Housekeeping questions to ignore
    housekeeping_ids = ['1761784371796', '1761784761563', '1761784922024', 
                       '1761786446770', '1761786362058']
    
    # Free text questions to ignore for now
    free_text_ids = ['1761789642623', '1761788166599', '1761788184302', 
                    '1761789275539', '1761844947638']
    
    print("\n" + "="*80)
    print("SURVEY ANALYSIS REPORT")
    print("="*80)
    
    # Process Likert scale questions
    print("\n\nLIKERT SCALE QUESTIONS (1-5 Scale)")
    print("="*80)
    
    likert_results = []
    for q_id, q_text in likert_questions.items():
        for question in data:
            if question['id'] == q_id:
                stats = calculate_likert_stats(question['answers'], q_text)
                if stats:
                    likert_results.append(stats)
                    print_likert_stats(stats)
                break
    
    # Process categorical questions
    print("\n\nCATEGORICAL/FREQUENCY QUESTIONS")
    print("="*80)
    
    categorical_results = []
    for q_id, q_text in categorical_questions.items():
        for question in data:
            if question['id'] == q_id:
                stats = analyze_categorical_question(question['answers'], q_text)
                if stats:
                    categorical_results.append(stats)
                    print_categorical_stats(stats)
                break
    
    # Summary statistics across all Likert questions
    print("\n\nSUMMARY ACROSS ALL LIKERT QUESTIONS")
    print("="*80)
    
    all_means = [s['mean'] for s in likert_results]
    print(f"Average of all means: {round(np.mean(all_means), 2)}")
    print(f"Questions with highest agreement (mean > 4.0):")
    for stats in likert_results:
        if stats['mean'] > 4.0:
            print(f"  - {stats['question'][:60]}... (Mean: {stats['mean']})")
    
    print(f"\nQuestions with lowest agreement (mean < 3.0):")
    for stats in likert_results:
        if stats['mean'] < 3.0:
            print(f"  - {stats['question'][:60]}... (Mean: {stats['mean']})")
    
    # Export to CSV for further analysis
    print("\n\nExporting results to CSV files...")
    
    # Create DataFrame for Likert results
    likert_df = pd.DataFrame([{
        'Question': s['question'],
        'N': s['n'],
        'Mean': s['mean'],
        'Median': s['median'],
        'Mode': s['mode'],
        'Std_Dev': s['std_dev'],
        'Min': s['min'],
        'Max': s['max']
    } for s in likert_results])
    
    likert_df.to_csv('likert_results.csv', index=False)
    print("Likert scale results saved to 'likert_results.csv'")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

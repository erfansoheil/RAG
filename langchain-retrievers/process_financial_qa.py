#!/usr/bin/env python3
"""
Process financial_qa.jsonl data for multi-query retriever evaluation.
Creates dataset with 69 companies, deduplicated contexts, and combined Q&A pairs.
"""

import json
import pandas as pd
from collections import defaultdict
import re
from difflib import SequenceMatcher

def is_continuation(context1, context2, threshold=0.8):
    """
    Check if context1 is a continuation of context2 or vice versa
    Returns True if one is a continuation of the other
    """
    # Check if one is a substring of the other
    if context1 in context2 or context2 in context1:
        return True
    
    # Check for high similarity (might be continuation with slight variations)
    similarity = SequenceMatcher(None, context1, context2).ratio()
    return similarity > threshold

def deduplicate_contexts(contexts):
    """
    Remove duplicate contexts and handle continuations
    Keep the longest context when one is a continuation of another
    """
    if len(contexts) <= 1:
        return contexts
    
    # Sort by length (longest first) to prioritize longer contexts
    sorted_contexts = sorted(set(contexts), key=len, reverse=True)
    
    unique_contexts = []
    for context in sorted_contexts:
        is_duplicate = False
        for existing in unique_contexts:
            if is_continuation(context, existing):
                # If current context is longer, replace the existing one
                if len(context) > len(existing):
                    unique_contexts.remove(existing)
                    unique_contexts.append(context)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_contexts.append(context)
    
    return unique_contexts

def main():
    # Load the financial QA data
    print("Loading financial QA data...")
    data = []
    with open('/home/erfan/TORUS/personal_git/RAG/langchain-retrievers/data/financial_qa.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Total entries: {len(data)}")
    
    # Group data by company
    print("Grouping data by company...")
    company_data = defaultdict(lambda: {
        'contexts': [],
        'questions': [],
        'answers': []
    })
    
    for entry in data:
        company = entry['company_name']
        company_data[company]['contexts'].append(entry['context'])
        company_data[company]['questions'].append(entry['question'])
        company_data[company]['answers'].append(entry['answer'])
    
    print(f"Grouped data for {len(company_data)} companies")
    
    # Process each company to deduplicate contexts and create final dataset
    print("Processing companies and deduplicating contexts...")
    processed_data = []
    
    for company, data_dict in company_data.items():
        # Deduplicate contexts
        unique_contexts = deduplicate_contexts(data_dict['contexts'])
        
        # Concatenate all unique contexts
        combined_context = ' '.join(unique_contexts)
        
        # Create the final entry
        processed_entry = {
            'company_name': company,
            'context': combined_context,
            'questions': data_dict['questions'],
            'answers': data_dict['answers']
        }
        
        processed_data.append(processed_entry)
        
        print(f"Company: {company}")
        print(f"  Original contexts: {len(data_dict['contexts'])}")
        print(f"  Unique contexts: {len(unique_contexts)}")
        print(f"  Questions: {len(data_dict['questions'])}")
        print(f"  Combined context length: {len(combined_context)}")
        print()
    
    # Save the processed data
    output_file = '/home/erfan/TORUS/personal_git/RAG/langchain-retrievers/data/processed_financial_qa_for_mqr.json'
    print(f"Saving processed data to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Create a summary of the processing
    summary = {
        'total_companies': len(processed_data),
        'total_original_entries': len(data),
        'average_questions_per_company': sum(len(entry['questions']) for entry in processed_data) / len(processed_data),
        'average_context_length': sum(len(entry['context']) for entry in processed_data) / len(processed_data),
        'companies_with_most_questions': sorted(
            [(entry['company_name'], len(entry['questions'])) for entry in processed_data],
            key=lambda x: x[1], reverse=True
        )[:10]
    }
    
    print("\nProcessing Summary:")
    print(f"Total companies: {summary['total_companies']}")
    print(f"Total original entries: {summary['total_original_entries']}")
    print(f"Average questions per company: {summary['average_questions_per_company']:.2f}")
    print(f"Average context length: {summary['average_context_length']:.0f} characters")
    print(f"Top 10 companies by question count:")
    for company, count in summary['companies_with_most_questions']:
        print(f"  {company}: {count} questions")
    
    print(f"\nSample processed entry:")
    print(json.dumps(processed_data[0], indent=2))
    
    print(f"\nData processing complete! Output saved to: {output_file}")

if __name__ == "__main__":
    main()

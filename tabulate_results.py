import json
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare test results with reference data')
    parser.add_argument('athene_file', help='Path to the Athene results file')
    parser.add_argument('reference_file', help='Path to the reference results file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load both JSON files
    try:
        with open(args.athene_file, 'r') as f:
            data = json.load(f)
        with open(args.reference_file, 'r') as f:
            reference = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return
        
    # Get the actual test cases
    current_data = data['per_scenario_results']
    reference_data = reference['per_scenario_results']
    
    # Get results with null traceback from current data
    null_traceback_results = [
        {'name': case['name'], 'similarity': case['similarity']}
        for case in current_data
        if case.get('traceback') is None
    ]
    
    # Sort by similarity
    null_traceback_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Create reference lookup dictionary
    reference_lookup = {case['name']: case['similarity'] for case in reference_data}
    
    # Calculate totals
    current_total = 0
    reference_total = 0
    reference_count = 0
    
    for result in null_traceback_results:
        name = result['name']
        current_sim = result['similarity']
        ref_sim = reference_lookup.get(name, "N/A")
        current_total += current_sim
        if ref_sim != "N/A":
            reference_total += ref_sim
            reference_count += 1
    
    # Calculate and print averages
    current_average = current_total / len(null_traceback_results)
    reference_average = reference_total / reference_count if reference_count > 0 else "N/A"
    
    print("\nAverages:")
    print("-" * 120)
    print(f"Athene-Agent average similarity: {current_average:.4f}")
    print(f"GPT4o average similarity: {reference_average if reference_average == 'N/A' else f'{reference_average:.4f}'}")
    print(f"\nTotal test cases with successful run: {len(null_traceback_results)}")
    print(f"Test cases used in reference: {reference_count}")

if __name__ == "__main__":
    main()
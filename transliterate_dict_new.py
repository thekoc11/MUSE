#!/usr/bin/env python3
# Script to transliterate Hindi words in dictionary from Devanagari to Latin script

from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from tqdm import tqdm
import os
import sys
import argparse

def is_devanagari(text):
    """Check if text contains Devanagari characters"""
    for char in text:
        # Unicode range for Devanagari
        if '\u0900' <= char <= '\u097F':
            return True
    return False

def transliterate_dictionary(input_path, output_path):
    """
    Read a tab-separated dictionary file with English and Hindi words,
    transliterate the Hindi words from Devanagari to Latin script,
    and write to a new file.
    
    Args:
        input_path: Path to the original dictionary file
        output_path: Path to write the transliterated dictionary
    """
    print(f"Transliterating dictionary from {input_path} to {output_path}")
    
    success_count = 0
    total_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        # Count lines for progress bar
        total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
        
        # Process each line
        for line in tqdm(infile, total=total_lines, desc="Transliterating"):
            total_count += 1
            
            # Skip empty lines
            if not line.strip():
                outfile.write('\n')
                continue
            
            # Split line into English and Hindi words
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"Warning: Unexpected format in line: {line}")
                outfile.write(line)
                continue
            
            english, hindi = parts
            
            # Check if the word contains Devanagari script
            if is_devanagari(hindi):
                # Transliterate Hindi from Devanagari to Latin (IAST)
                try:
                    transliterated = transliterate(hindi, sanscript.DEVANAGARI, sanscript.IAST)
                    success_count += 1
                except Exception as e:
                    print(f"Error transliterating: {hindi} - {str(e)}")
                    transliterated = hindi  # Keep original if transliteration fails
            else:
                # If not Devanagari, keep as is
                transliterated = hindi
            
            # Write English and transliterated Hindi to output file
            outfile.write(f"{english}\t{transliterated}\n")
    
    print(f"Transliteration completed, saved to {output_path}")
    print(f"Successfully transliterated {success_count} out of {total_count} entries")
    return True

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Transliterate Hindi dictionary from Devanagari to Latin script.')
    parser.add_argument('input_dict', help='Path to the input dictionary file')
    parser.add_argument('--output', '-o', help='Path to the output dictionary file (default: input_file with _latin suffix)')
    args = parser.parse_args()
    
    # Get input path from command line arguments
    input_path = args.input_dict
    
    # Generate default output path if not specified
    if args.output:
        output_path = args.output
    else:
        # Create output path based on input path
        input_basename = os.path.basename(input_path)
        input_dir = os.path.dirname(input_path)
        output_basename = input_basename.replace('.txt', '_latin.txt')
        if output_basename == input_basename:  # If no .txt extension
            output_basename = input_basename + '_latin'
        output_path = os.path.join(input_dir, output_basename)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Print debug information
    print("Python version:", sys.version)
    print("indic-transliteration version:", sanscript.__version__ if hasattr(sanscript, "__version__") else "Unknown")
    
    # Verify with a simple test
    test_text = "नमस्ते"
    test_transliterated = transliterate(test_text, sanscript.DEVANAGARI, sanscript.IAST)
    print(f"Test transliteration: {test_text} -> {test_transliterated}")
    
    # Also test with HK (Harvard-Kyoto) which is closer to standard Latin
    test_transliterated_hk = transliterate(test_text, sanscript.DEVANAGARI, sanscript.HK)
    print(f"Test HK transliteration: {test_text} -> {test_transliterated_hk}")
    
    # Try another output scheme - ITRANS is easier to read for non-Sanskrit readers
    test_transliterated_itrans = transliterate(test_text, sanscript.DEVANAGARI, sanscript.ITRANS)
    print(f"Test ITRANS transliteration: {test_text} -> {test_transliterated_itrans}")
    
    # Ask user which scheme to use
    print("\nAvailable transliteration schemes:")
    print("1. IAST (namaste)")
    print("2. ITRANS (namaste)")
    print("3. Harvard-Kyoto (namaste)")
    
    scheme_choice = input("Choose a scheme (1-3) or press Enter for default (ITRANS): ")
    
    output_scheme = sanscript.ITRANS  # Default
    if scheme_choice == "1":
        output_scheme = sanscript.IAST
        # output_path = output_path.replace(".txt", "_iast.txt")
        print(f"Using IAST scheme, output will be saved to {output_path}")
    elif scheme_choice == "2" or scheme_choice == "":
        output_scheme = sanscript.ITRANS
        output_path = output_path.replace(".txt", "_itrans.txt")
        print(f"Using ITRANS scheme, output will be saved to {output_path}")
    elif scheme_choice == "3":
        output_scheme = sanscript.HK
        output_path = output_path.replace(".txt", "_hk.txt")
        print(f"Using Harvard-Kyoto scheme, output will be saved to {output_path}")
    
    # Transliterate the dictionary with the chosen scheme
    def transliterate_with_scheme(input_path, output_path, scheme):
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
            success_count = 0
            total_count = 0
            
            for line in tqdm(infile, total=total_lines, desc=f"Transliterating to {scheme}"):
                total_count += 1
                
                if not line.strip():
                    outfile.write('\n')
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    outfile.write(line)
                    continue
                
                english, hindi = parts
                
                if is_devanagari(hindi):
                    try:
                        transliterated = transliterate(hindi, sanscript.DEVANAGARI, scheme)
                        success_count += 1
                    except Exception as e:
                        print(f"Error transliterating: {hindi} - {str(e)}")
                        transliterated = hindi
                else:
                    transliterated = hindi
                
                outfile.write(f"{english}\t{transliterated}\n")
            
            print(f"Successfully transliterated {success_count} out of {total_count} entries to {scheme}")
    
    transliterate_with_scheme(input_path, output_path, output_scheme) 
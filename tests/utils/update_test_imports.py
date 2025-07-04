#!/usr/bin/env python3
"""
Script to update all test files with proper import paths
"""

import os
import re

def update_test_file(filepath):
    """Update a test file to have correct import paths for tests directory"""
    print(f"Updating {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already has the sys.path addition
    if 'sys.path.insert(0, os.path.join(os.path.dirname(__file__), \'..\'))' in content:
        print(f"  {filepath} already updated")
        return
    
    # Find the first import line and add sys.path before it
    lines = content.split('\n')
    
    # Find where to insert the path modification
    insert_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_index = i
            break
    
    if insert_index is None:
        print(f"  No imports found in {filepath}")
        return
    
    # Add the necessary imports and path modification
    new_lines = []
    
    # Copy content before first import
    new_lines.extend(lines[:insert_index])
    
    # Add sys and os imports if not present
    has_sys = any('import sys' in line for line in lines[:insert_index])
    has_os = any('import os' in line for line in lines[:insert_index])
    
    if not has_os:
        new_lines.append('import os')
    if not has_sys:
        new_lines.append('import sys')
    
    # Add empty line if we added imports
    if not has_os or not has_sys:
        new_lines.append('')
    
    # Add the path modification
    new_lines.append('# Add parent directory to path for imports')
    new_lines.append('sys.path.insert(0, os.path.join(os.path.dirname(__file__), \'..\'))')
    new_lines.append('')
    
    # Add remaining content
    new_lines.extend(lines[insert_index:])
    
    # Write back the updated content
    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"  Updated {filepath}")

def main():
    tests_dir = '/mnt/workspace/nanoGPT/tests'
    
    # Get all Python files in tests directory
    test_files = []
    for filename in os.listdir(tests_dir):
        if filename.endswith('.py') and filename != 'update_test_imports.py':
            test_files.append(os.path.join(tests_dir, filename))
    
    print(f"Found {len(test_files)} test files to update:")
    for filepath in test_files:
        print(f"  {os.path.basename(filepath)}")
    
    print("\nUpdating files...")
    for filepath in test_files:
        update_test_file(filepath)
    
    print("\n✅ All test files updated!")

if __name__ == '__main__':
    main()

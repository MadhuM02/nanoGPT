#!/usr/bin/env python3
"""
Fix import paths for all test files after reorganization
"""

import os
import re
import glob

def fix_import_path(file_path):
    """Fix the sys.path.insert line in a test file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Determine how many levels up we need to go
    relative_path = os.path.relpath(file_path, '/mnt/workspace/nanoGPT')
    depth = len(relative_path.split('/')) - 1  # -1 because we don't count the file itself
    
    if depth == 1:
        # tests/file.py -> go up 1 level
        levels_up = '..'
    elif depth == 2:
        # tests/subdir/file.py -> go up 2 levels  
        levels_up = os.path.join('..', '..')
    else:
        # Handle deeper nesting if needed
        levels_up = os.path.join(*(['..'] * depth))
    
    # Pattern to match the sys.path.insert line
    pattern = r"sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\),\s*['\"]\.\.['\"]\s*\)\)"
    replacement = f"sys.path.insert(0, os.path.join(os.path.dirname(__file__), '{levels_up}'))"
    
    # Also handle patterns with multiple '..' arguments
    pattern2 = r"sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\),\s*['\"]\.\.['\"]\s*,\s*['\"]\.\.['\"]\s*\)\)"
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print(f"Fixed {file_path} (depth {depth})")
    elif re.search(pattern2, content):
        content = re.sub(pattern2, replacement, content)
        print(f"Fixed {file_path} (depth {depth}) - multi-level")
    elif 'sys.path.insert' in content and 'dirname(__file__)' in content:
        # Handle any other variations
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'sys.path.insert' in line and 'dirname(__file__)' in line:
                lines[i] = f"sys.path.insert(0, os.path.join(os.path.dirname(__file__), '{levels_up}'))"
                break
        content = '\n'.join(lines)
        print(f"Fixed {file_path} (depth {depth}) - custom pattern")
    
    with open(file_path, 'w') as f:
        f.write(content)

def main():
    """Fix all test files"""
    test_dir = '/mnt/workspace/nanoGPT/tests'
    
    # Find all Python test files in subdirectories
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py') and (file.startswith('test_') or file.startswith('debug_') or file.startswith('analyze_') or file.startswith('minimal_')):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} test files to fix:")
    for file_path in test_files:
        fix_import_path(file_path)
    
    print("✅ All import paths fixed!")

if __name__ == '__main__':
    main()

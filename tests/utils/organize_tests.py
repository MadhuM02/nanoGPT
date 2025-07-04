#!/usr/bin/env python3
"""
Test organization summary and verification
"""

import os
import sys

def check_file_structure():
    """Check that all files are properly organized"""
    print("🔍 Checking file organization...")
    
    root_dir = os.path.dirname(os.path.dirname(__file__))
    tests_dir = os.path.dirname(__file__)
    
    # Check that test files are in tests directory
    test_files_in_tests = []
    for f in os.listdir(tests_dir):
        if f.startswith('test_') or f == 'quick_test.py':
            test_files_in_tests.append(f)
    
    # Check that no test files remain in root
    test_files_in_root = []
    for f in os.listdir(root_dir):
        if f.startswith('test_') or f == 'quick_test.py':
            test_files_in_root.append(f)
    
    print(f"✅ Test files in tests/: {len(test_files_in_tests)}")
    for f in sorted(test_files_in_tests):
        print(f"   {f}")
    
    if test_files_in_root:
        print(f"⚠️  Test files still in root: {test_files_in_root}")
    else:
        print("✅ No test files in root directory")
    
    return len(test_files_in_root) == 0

def check_imports():
    """Check that test files have correct import paths"""
    print("\n🔍 Checking import paths...")
    
    tests_dir = os.path.dirname(__file__)
    issues = []
    
    for filename in os.listdir(tests_dir):
        if (filename.endswith('.py') and 
            filename not in ['organize_tests.py', 'update_test_imports.py', 'run_tests.py'] and
            not filename.startswith('.')):
            filepath = os.path.join(tests_dir, filename)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Check if it has the path modification
            if 'sys.path.insert(0, os.path.join(os.path.dirname(__file__), \'..\'))' not in content:
                issues.append(f"{filename}: Missing path modification")
    
    if issues:
        print("⚠️  Import issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ All test files have correct import paths")
        return True

def main():
    print("nanoGPT Test Organization Verification")
    print("=" * 50)
    
    structure_ok = check_file_structure()
    imports_ok = check_imports()
    
    print("\n📊 Summary:")
    print(f"  File structure: {'✅ OK' if structure_ok else '❌ Issues'}")
    print(f"  Import paths: {'✅ OK' if imports_ok else '❌ Issues'}")
    
    if structure_ok and imports_ok:
        print("\n🎉 All tests are properly organized!")
        print("\nUsage:")
        print("  cd tests")
        print("  python run_tests.py          # Run all tests")
        print("  python test_moe.py           # Run specific test")
        print("  python quick_test.py         # Quick smoke test")
        return 0
    else:
        print("\n💥 Issues found. Please check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

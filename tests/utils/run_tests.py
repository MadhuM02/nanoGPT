#!/usr/bin/env python3
"""
Test runner for nanoGPT MoE tests
Runs all tests and provides a summary
"""

import os
import sys
import subprocess
import time

def find_test_files():
    """Find all test files in the organized directory structure"""
    test_files = []
    test_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to tests/ directory
    
    # Search in all subdirectories for test files
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py') and (file.startswith('test_') or file.startswith('minimal_')):
                # Skip the fix_imports.py and other utility scripts
                if file not in ['fix_imports.py', 'run_tests.py', 'organize_tests.py', 'update_test_imports.py']:
                    test_files.append(os.path.join(root, file))
    
    return sorted(test_files)

def run_test(test_file):
    """Run a single test file and return results"""
    print(f"\n{'='*60}")
    print(f"Running {os.path.relpath(test_file)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_file} PASSED ({duration:.1f}s)")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            return True, duration
        else:
            print(f"❌ {test_file} FAILED ({duration:.1f}s)")
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"STDERR:\n{result.stderr}")
            return False, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} TIMEOUT (>120s)")
        return False, 120
    except Exception as e:
        print(f"💥 {test_file} ERROR: {e}")
        return False, 0

def main():
    print("nanoGPT Test Suite")
    print("=" * 40)
    
    # Find all test files in organized structure
    test_files = find_test_files()
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        rel_path = os.path.relpath(test_file)
        print(f"  {rel_path}")
    
    # Run tests
    results = {}
    total_time = 0
    
    for test_file in test_files:
        passed, duration = run_test(test_file)
        results[test_file] = passed
        total_time += duration
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"Tests run: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Total time: {total_time:.1f}s")
    
    print(f"\nDetailed results:")
    for test_file, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        rel_path = os.path.relpath(test_file)
        print(f"  {rel_path}: {status}")
    
    if passed_count == total_count:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 {total_count - passed_count} test(s) failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())

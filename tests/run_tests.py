#!/usr/bin/env python3
"""
Test runner for nanoGPT MoE tests
Runs all tests and provides a summary
"""

import os
import sys
import subprocess
import time

def run_test(test_file):
    """Run a single test file and return results"""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
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
    print("nanoGPT MoE Test Suite")
    print("=" * 40)
    
    tests_dir = os.path.dirname(__file__)
    
    # Find all test files
    test_files = []
    for filename in sorted(os.listdir(tests_dir)):
        if filename.startswith('test_') and filename.endswith('.py'):
            test_files.append(filename)
    
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  {test_file}")
    
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
        print(f"  {test_file}: {status}")
    
    if passed_count == total_count:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 {total_count - passed_count} test(s) failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())

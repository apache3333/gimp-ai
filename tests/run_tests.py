#!/usr/bin/env python3
"""
Test runner for GIMP AI Plugin coordinate transformations.

This script runs all coordinate transformation tests and reports results.
It can be run without GIMP installed since it only tests pure Python functions.
"""

import importlib
import sys
import os

def main():
    """Run all tests and report results."""
    print("GIMP AI Plugin - Coordinate Transformation Test Suite")
    print("=" * 60)
    
    # Ensure we can import test modules
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    
    # (description, module, entry point) - each entry point returns True on success
    suites = [
        ("coordinate transformation", "test_coordinate_transformations", "run_all_tests"),
        ("shape-aware function", "test_shape_aware_functions", "run_all_tests"),
        ("aspect ratio extension", "test_aspect_ratio_extension", "run_test"),
        ("integration", "test_integration", "run_all_tests"),
        ("provider abstraction", "test_providers", "run_all_tests"),
    ]

    # Import and run all test suites
    try:
        failed = []

        for description, module_name, entry_point in suites:
            print(f"\nRunning {description} tests...")
            module = importlib.import_module(module_name)
            if not getattr(module, entry_point)():
                failed.append(description)

        if not failed:
            print("\n🎉 All tests completed successfully!")
            print("The coordinate transformation system is mathematically correct.")
            print("The provider abstraction builds and parses requests correctly.")
            return 0
        else:
            print("\n❌ Some tests failed.")
            print("Failed suites: " + ", ".join(failed))
            return 1
            
    except ImportError as e:
        print(f"❌ Failed to import test modules: {e}")
        print("Make sure coordinate_utils.py and ai_providers.py are in the parent directory.")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
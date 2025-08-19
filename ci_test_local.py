#!/usr/bin/env python3
"""
Local CI simulation to test what might be failing
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success/failure"""
    print(f"\n🔍 Testing: {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {description}")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {description} - {e}")
        return False

def main():
    print("🧪 AdaptML CI Local Testing")
    print("=" * 50)
    
    # Change to repo directory
    os.chdir('/Users/krispetersen/adaptml_repo')
    
    tests = [
        ("pip install -r requirements.txt", "Install requirements"),
        ("pip install pytest", "Install pytest"),
        ("python -c 'import adaptml; print(\"Import successful\")'", "Test import"),
        ("python -c 'import adaptml; print(f\"Email: {adaptml.__email__}\"); print(f\"Website: {adaptml.__website__}\")'", "Test contact info"),
        ("python -m pytest tests/ -v", "Run pytest"),
        ("python test_adaptml.py", "Run direct test"),
        ("python -c 'from adaptml import quickstart; quickstart()'", "Run quickstart demo"),
    ]
    
    passed = 0
    total = len(tests)
    
    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CI should work.")
    else:
        print("⚠️  Some tests failed. CI might have issues.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

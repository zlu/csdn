#!/usr/bin/env python3
"""
Test Setup Script for Qwen-TTS Sichuan Demo
===========================================

This script tests the setup and dependencies for the Qwen-TTS demo.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import dashscope
        print("✅ dashscope imported successfully")
    except ImportError as e:
        print(f"❌ dashscope import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    try:
        import pygame
        print("✅ pygame imported successfully")
    except ImportError as e:
        print(f"⚠️ pygame import failed: {e}")
        print("   Audio playback will be disabled, but TTS will still work")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"⚠️ python-dotenv import failed: {e}")
        print("   .env files will not be loaded, but environment variables will still work")
    
    return True

def test_env_file():
    """Test if .env file exists and can be loaded."""
    print("\n📄 Testing .env file...")
    
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("⚠️ python-dotenv not available, skipping .env file test")
        return True
    
    env_files = [".env", "../.env", "../../.env"]
    env_found = False
    
    for env_file in env_files:
        env_path = Path(env_file)
        if env_path.exists():
            print(f"✅ Found .env file: {env_path.absolute()}")
            env_found = True
            
            # Try to load it
            try:
                load_dotenv(env_path)
                print(f"✅ Successfully loaded .env file: {env_path}")
                
                # Check if it contains the API key
                api_key = os.getenv("DASHSCOPE_API_KEY")
                if api_key:
                    print(f"✅ API key found in .env file (starts with: {api_key[:8]}...)")
                else:
                    print("⚠️ No DASHSCOPE_API_KEY found in .env file")
                
                break
            except Exception as e:
                print(f"❌ Failed to load .env file: {e}")
    
    if not env_found:
        print("ℹ️ No .env file found in current or parent directories")
        print("   You can create one with: echo 'DASHSCOPE_API_KEY=your_key_here' > .env")
    
    return True

def test_api_key():
    """Test if API key is set."""
    print("\n🔑 Testing API key...")
    
    # Try multiple possible environment variable names
    possible_keys = [
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY", 
        "QWEN_TTS_API_KEY",
        "ALIBABA_API_KEY"
    ]
    
    api_key = None
    key_name = None
    
    for key in possible_keys:
        api_key = os.getenv(key)
        if api_key:
            key_name = key
            break
    
    if api_key:
        print(f"✅ API key found using environment variable: {key_name}")
        print(f"   Key starts with: {api_key[:8]}...")
        
        # Validate format
        if api_key.startswith("sk-"):
            print("✅ API key format looks correct (starts with 'sk-')")
        else:
            print("⚠️ API key doesn't start with 'sk-'. This might not be a valid DashScope API key.")
        
        return True
    else:
        print("❌ No API key found in environment variables")
        print("   Available methods to set API key:")
        print("   1. Environment variable: export DASHSCOPE_API_KEY='your_key_here'")
        print("   2. .env file: Create .env with DASHSCOPE_API_KEY=your_key_here")
        print("   3. Alternative names: QWEN_API_KEY, QWEN_TTS_API_KEY, ALIBABA_API_KEY")
        return False

def test_directories():
    """Test if required directories exist or can be created."""
    print("\n📁 Testing directories...")
    
    output_dir = Path("output")
    try:
        output_dir.mkdir(exist_ok=True)
        print(f"✅ Output directory ready: {output_dir.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Failed to create output directory: {e}")
        return False

def test_network():
    """Test network connectivity."""
    print("\n🌐 Testing network connectivity...")
    
    try:
        import requests
        response = requests.get("https://httpbin.org/get", timeout=5)
        if response.status_code == 200:
            print("✅ Network connectivity OK")
            return True
        else:
            print(f"❌ Network test failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Qwen-TTS Setup Test")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Environment File", test_env_file),
        ("API Key", test_api_key),
        ("Directories", test_directories),
        ("Network", test_network),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! You're ready to run the demo.")
        print("   Run: python qwen_tts_sichuan_demo.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues before running the demo.")
        print("\n💡 Quick fixes:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Set API key: export DASHSCOPE_API_KEY='your_key_here'")
        print("   3. Or create .env file: echo 'DASHSCOPE_API_KEY=your_key_here' > .env")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
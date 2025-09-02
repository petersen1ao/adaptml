#!/usr/bin/env python3
"""
Debug obfuscated malware detection issue
"""

import base64
import binascii
import urllib.parse
import re

def test_deobfuscation():
    """Test the deobfuscation process step by step"""
    
    # Original malicious code
    malicious_code = "import os; os.system('rm -rf /')"
    print(f"🎯 Original malicious code: {malicious_code}")
    print()
    
    # Layer 1: Base64 encode
    layer1 = base64.b64encode(malicious_code.encode()).decode()
    print(f"Layer 1 (Base64): {layer1}")
    
    # Layer 2: Hex encode the base64
    layer2 = layer1.encode().hex()
    print(f"Layer 2 (Hex): {layer2}")
    
    # Layer 3: URL encode the hex
    layer3 = ''.join(f'%{ord(c):02x}' for c in layer2)
    print(f"Layer 3 (URL): {layer3[:100]}...")
    print()
    
    # Now test deobfuscation process
    print("🔍 Testing deobfuscation process:")
    current_data = layer3
    
    # Step 1: URL decode
    print(f"Before URL decode: {current_data[:50]}...")
    if '%' in current_data:
        try:
            decoded_url = urllib.parse.unquote(current_data)
            print(f"After URL decode: {decoded_url[:50]}...")
            current_data = decoded_url
        except Exception as e:
            print(f"URL decode failed: {e}")
    
    # Step 2: Hex decode
    print(f"Before hex decode: {current_data[:50]}...")
    hex_pattern = r'[0-9a-fA-F]{20,}'
    hex_matches = re.findall(hex_pattern, current_data)
    if hex_matches:
        for match in hex_matches:
            if len(match) % 2 == 0:  # Valid hex length
                try:
                    decoded_hex = bytes.fromhex(match).decode('utf-8', errors='ignore')
                    if decoded_hex.isprintable():
                        print(f"Hex decoded: {decoded_hex}")
                        current_data = current_data.replace(match, decoded_hex)
                        break
                except Exception as e:
                    print(f"Hex decode failed: {e}")
    
    # Step 3: Base64 decode
    print(f"Before base64 decode: {current_data[:50]}...")
    base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
    base64_matches = re.findall(base64_pattern, current_data)
    if base64_matches:
        for match in base64_matches:
            try:
                decoded_base64 = base64.b64decode(match).decode('utf-8', errors='ignore')
                print(f"Base64 decoded: {decoded_base64}")
                current_data = current_data.replace(match, decoded_base64)
                break
            except Exception as e:
                print(f"Base64 decode failed: {e}")
    
    print()
    print(f"🎯 Final deobfuscated content: {current_data}")
    
    # Check if malicious content is revealed
    malicious_keywords = ['rm -rf', 'os.system', 'import os', 'system(']
    found_threats = []
    for keyword in malicious_keywords:
        if keyword.lower() in current_data.lower():
            found_threats.append(keyword)
    
    if found_threats:
        print(f"✅ SUCCESS: Found malicious patterns: {found_threats}")
        return True
    else:
        print("❌ FAILED: No malicious patterns detected")
        return False

if __name__ == "__main__":
    success = test_deobfuscation()
    print(f"\n🎯 Obfuscated malware detection: {'SUCCESS' if success else 'FAILED'}")

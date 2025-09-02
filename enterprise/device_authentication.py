# 🔐 **AdaptML Device Authentication & Licensing System**
# *Secure device registration and package enforcement for business packages*

import hashlib
import secrets
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import hmac
import base64
import platform
import uuid
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DeviceFingerprint:
    """Unique device identification"""
    device_id: str
    hostname: str
    platform_info: str
    cpu_count: int
    total_memory: int
    disk_serial: str
    network_mac: str
    bios_uuid: str
    created_at: datetime
    
@dataclass
class LicensePackage:
    """AdaptML license package configuration"""
    package_id: str
    package_name: str
    device_limit: int
    is_lifetime: bool
    monthly_price: float
    lifetime_price: Optional[float]
    features: List[str]
    
@dataclass
class RegisteredDevice:
    """Device registered to a license"""
    device_id: str
    license_key: str
    device_fingerprint: DeviceFingerprint
    registration_date: datetime
    last_seen: datetime
    is_active: bool
    
@dataclass
class AdaptMLLicense:
    """Complete AdaptML license with device tracking"""
    license_key: str
    customer_id: str
    package: LicensePackage
    registered_devices: List[RegisteredDevice]
    issue_date: datetime
    expiry_date: Optional[datetime]  # None for lifetime
    is_active: bool
    usage_stats: Dict[str, any]

class DeviceFingerprintGenerator:
    """Generate unique, stable device fingerprints"""
    
    @staticmethod
    def generate_device_fingerprint() -> DeviceFingerprint:
        """Create comprehensive device fingerprint"""
        try:
            # Gather system information
            hostname = platform.node()
            platform_info = f"{platform.system()}-{platform.release()}-{platform.machine()}"
            cpu_count = psutil.cpu_count(logical=False)
            total_memory = psutil.virtual_memory().total
            
            # Network MAC address (first active interface)
            network_mac = "unknown"
            for interface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == psutil.AF_LINK and addr.address != "00:00:00:00:00:00":
                        network_mac = addr.address
                        break
                if network_mac != "unknown":
                    break
            
            # Disk serial (first disk)
            disk_serial = "unknown"
            try:
                for disk in psutil.disk_partitions():
                    if 'fixed' in disk.opts:
                        # Get disk serial number (platform specific)
                        disk_serial = f"disk-{hash(disk.device) % 1000000:06d}"
                        break
            except:
                disk_serial = "disk-unknown"
            
            # BIOS UUID (platform specific)
            try:
                if platform.system() == "Darwin":  # macOS
                    import subprocess
                    result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                          capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if 'Hardware UUID' in line:
                            bios_uuid = line.split(':')[1].strip()
                            break
                    else:
                        bios_uuid = str(uuid.uuid4())
                elif platform.system() == "Windows":
                    import subprocess
                    result = subprocess.run(['wmic', 'csproduct', 'get', 'UUID'], 
                                          capture_output=True, text=True)
                    lines = result.stdout.strip().split('\n')
                    bios_uuid = lines[1].strip() if len(lines) > 1 else str(uuid.uuid4())
                else:  # Linux
                    try:
                        with open('/sys/class/dmi/id/product_uuid', 'r') as f:
                            bios_uuid = f.read().strip()
                    except:
                        bios_uuid = str(uuid.uuid4())
            except:
                bios_uuid = str(uuid.uuid4())
            
            # Create stable device ID
            device_components = f"{hostname}-{platform_info}-{cpu_count}-{total_memory}-{network_mac}-{disk_serial}-{bios_uuid}"
            device_id = hashlib.sha256(device_components.encode()).hexdigest()[:16]
            
            return DeviceFingerprint(
                device_id=device_id,
                hostname=hostname,
                platform_info=platform_info,
                cpu_count=cpu_count,
                total_memory=total_memory,
                disk_serial=disk_serial,
                network_mac=network_mac,
                bios_uuid=bios_uuid,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating device fingerprint: {e}")
            # Fallback to basic fingerprint
            fallback_id = hashlib.sha256(f"{platform.node()}-{time.time()}".encode()).hexdigest()[:16]
            return DeviceFingerprint(
                device_id=fallback_id,
                hostname=platform.node(),
                platform_info=platform.system(),
                cpu_count=1,
                total_memory=0,
                disk_serial="unknown",
                network_mac="unknown",
                bios_uuid="unknown",
                created_at=datetime.now()
            )

class AdaptMLLicenseManager:
    """Comprehensive license and device management for AdaptML"""
    
    def __init__(self, database_path: str = "adaptml_licenses.db"):
        self.database_path = database_path
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # AdaptML Package Definitions
        self.packages = {
            "entry": LicensePackage(
                package_id="entry",
                package_name="Entry",
                device_limit=1,
                is_lifetime=False,
                monthly_price=299.0,
                lifetime_price=None,  # Contact for pricing
                features=["Basic AI Acceleration", "1 Device/Project", "Email Support"]
            ),
            "pro": LicensePackage(
                package_id="pro",
                package_name="Pro",
                device_limit=5,
                is_lifetime=False,
                monthly_price=1499.0,
                lifetime_price=None,  # Contact for pricing
                features=["Advanced AI Acceleration", "5 Devices/Projects", "Priority Support", "API Access"]
            ),
            "max": LicensePackage(
                package_id="max",
                package_name="Max",
                device_limit=20,
                is_lifetime=False,
                monthly_price=4999.0,
                lifetime_price=None,  # Contact for pricing
                features=["Enterprise AI Acceleration", "20 Devices/Projects", "Premium Support", "Full API Access", "Analytics Dashboard"]
            )
        }
        
        self._initialize_database()
        logger.info("🔐 AdaptML License Manager initialized")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for license data"""
        key_file = "adaptml_encryption.key"
        try:
            with open(key_file, 'rb') as f:
                return f.read()
        except FileNotFoundError:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            logger.info("🔑 Generated new encryption key")
            return key
    
    def _initialize_database(self):
        """Initialize SQLite database for license management"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Licenses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS licenses (
                    license_key TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    package_id TEXT NOT NULL,
                    issue_date TEXT NOT NULL,
                    expiry_date TEXT,
                    is_active BOOLEAN NOT NULL,
                    usage_stats TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Registered devices table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registered_devices (
                    device_id TEXT PRIMARY KEY,
                    license_key TEXT NOT NULL,
                    device_fingerprint TEXT NOT NULL,
                    registration_date TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL,
                    FOREIGN KEY (license_key) REFERENCES licenses (license_key)
                )
            ''')
            
            # License validation log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    license_key TEXT NOT NULL,
                    validation_result TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("📊 Database initialized successfully")
    
    def generate_license_key(self, customer_id: str, package_id: str, is_lifetime: bool = False) -> str:
        """Generate secure license key"""
        timestamp = int(time.time())
        components = f"{customer_id}-{package_id}-{timestamp}-{secrets.token_hex(8)}"
        
        if is_lifetime:
            components += "-LIFETIME"
        
        # Create HMAC signature
        signature = hmac.new(
            self.encryption_key,
            components.encode(),
            hashlib.sha256
        ).hexdigest()[:8]
        
        license_key = f"ADAPTML-{components}-{signature}".upper()
        return license_key
    
    def validate_license_key(self, license_key: str) -> bool:
        """Validate license key format and signature"""
        try:
            if not license_key.startswith("ADAPTML-"):
                return False
            
            parts = license_key[8:].split('-')
            if len(parts) < 5:
                return False
            
            # Reconstruct components without signature
            components = '-'.join(parts[:-1])
            provided_signature = parts[-1]
            
            # Calculate expected signature
            expected_signature = hmac.new(
                self.encryption_key,
                components.encode(),
                hashlib.sha256
            ).hexdigest()[:8].upper()
            
            return hmac.compare_digest(provided_signature, expected_signature)
            
        except Exception as e:
            logger.error(f"❌ License validation error: {e}")
            return False
    
    def create_license(self, customer_id: str, package_id: str, is_lifetime: bool = False) -> str:
        """Create new AdaptML license"""
        if package_id not in self.packages:
            raise ValueError(f"Invalid package ID: {package_id}")
        
        package = self.packages[package_id]
        license_key = self.generate_license_key(customer_id, package_id, is_lifetime)
        
        issue_date = datetime.now()
        expiry_date = None if is_lifetime else issue_date + timedelta(days=30)
        
        # Store in database
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO licenses (license_key, customer_id, package_id, issue_date, expiry_date, is_active, usage_stats)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                license_key,
                customer_id,
                package_id,
                issue_date.isoformat(),
                expiry_date.isoformat() if expiry_date else None,
                True,
                json.dumps({"devices_registered": 0, "total_activations": 0})
            ))
            conn.commit()
        
        logger.info(f"✅ Created license {license_key} for customer {customer_id}")
        return license_key
    
    def register_device(self, license_key: str, force_registration: bool = False) -> Tuple[bool, str]:
        """Register current device to license"""
        
        # Validate license key format
        if not self.validate_license_key(license_key):
            return False, "Invalid license key format"
        
        # Get license from database
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM licenses WHERE license_key = ? AND is_active = 1', (license_key,))
            license_row = cursor.fetchone()
            
            if not license_row:
                return False, "License not found or inactive"
            
            # Check expiry
            if license_row[4]:  # expiry_date
                expiry = datetime.fromisoformat(license_row[4])
                if datetime.now() > expiry:
                    return False, "License expired"
            
            package_id = license_row[2]
            package = self.packages[package_id]
            
            # Generate device fingerprint
            device_fingerprint = DeviceFingerprintGenerator.generate_device_fingerprint()
            
            # Check if device already registered
            cursor.execute('SELECT * FROM registered_devices WHERE device_id = ? AND license_key = ?', 
                          (device_fingerprint.device_id, license_key))
            existing_device = cursor.fetchone()
            
            if existing_device:
                # Update last seen
                cursor.execute('UPDATE registered_devices SET last_seen = ? WHERE device_id = ?', 
                              (datetime.now().isoformat(), device_fingerprint.device_id))
                conn.commit()
                return True, "Device already registered and updated"
            
            # Check device limit
            cursor.execute('SELECT COUNT(*) FROM registered_devices WHERE license_key = ? AND is_active = 1', 
                          (license_key,))
            active_devices = cursor.fetchone()[0]
            
            if active_devices >= package.device_limit and not force_registration:
                return False, f"Device limit reached ({active_devices}/{package.device_limit})"
            
            # Register new device
            cursor.execute('''
                INSERT INTO registered_devices (device_id, license_key, device_fingerprint, registration_date, last_seen, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                device_fingerprint.device_id,
                license_key,
                self.cipher.encrypt(json.dumps(asdict(device_fingerprint), default=str).encode()).decode(),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                True
            ))
            
            # Update license usage stats
            usage_stats = json.loads(license_row[6])
            usage_stats["devices_registered"] = active_devices + 1
            usage_stats["total_activations"] = usage_stats.get("total_activations", 0) + 1
            
            cursor.execute('UPDATE licenses SET usage_stats = ? WHERE license_key = ?', 
                          (json.dumps(usage_stats), license_key))
            
            conn.commit()
        
        logger.info(f"✅ Device {device_fingerprint.device_id} registered to license {license_key}")
        return True, "Device registered successfully"
    
    def validate_device_license(self, license_key: str) -> Tuple[bool, str, Dict]:
        """Validate if current device can use AdaptML with this license"""
        
        device_fingerprint = DeviceFingerprintGenerator.generate_device_fingerprint()
        
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Check license validity
            cursor.execute('SELECT * FROM licenses WHERE license_key = ? AND is_active = 1', (license_key,))
            license_row = cursor.fetchone()
            
            if not license_row:
                self._log_validation(device_fingerprint.device_id, license_key, "LICENSE_NOT_FOUND")
                return False, "License not found or inactive", {}
            
            # Check expiry
            if license_row[4]:  # expiry_date
                expiry = datetime.fromisoformat(license_row[4])
                if datetime.now() > expiry:
                    self._log_validation(device_fingerprint.device_id, license_key, "LICENSE_EXPIRED")
                    return False, "License expired", {}
            
            # Check device registration
            cursor.execute('SELECT * FROM registered_devices WHERE device_id = ? AND license_key = ? AND is_active = 1', 
                          (device_fingerprint.device_id, license_key))
            device_row = cursor.fetchone()
            
            if not device_row:
                self._log_validation(device_fingerprint.device_id, license_key, "DEVICE_NOT_REGISTERED")
                return False, "Device not registered to this license", {}
            
            # Update last seen
            cursor.execute('UPDATE registered_devices SET last_seen = ? WHERE device_id = ?', 
                          (datetime.now().isoformat(), device_fingerprint.device_id))
            conn.commit()
        
        package_id = license_row[2]
        package = self.packages[package_id]
        
        license_info = {
            "package_name": package.package_name,
            "device_limit": package.device_limit,
            "features": package.features,
            "is_lifetime": "LIFETIME" in license_key,
            "device_id": device_fingerprint.device_id
        }
        
        self._log_validation(device_fingerprint.device_id, license_key, "VALIDATION_SUCCESS")
        return True, "License valid", license_info
    
    def _log_validation(self, device_id: str, license_key: str, result: str):
        """Log validation attempt"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO validation_log (device_id, license_key, validation_result)
                VALUES (?, ?, ?)
            ''', (device_id, license_key, result))
            conn.commit()
    
    def get_license_info(self, license_key: str) -> Optional[Dict]:
        """Get comprehensive license information"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM licenses WHERE license_key = ?', (license_key,))
            license_row = cursor.fetchone()
            
            if not license_row:
                return None
            
            # Get registered devices
            cursor.execute('SELECT * FROM registered_devices WHERE license_key = ? AND is_active = 1', (license_key,))
            device_rows = cursor.fetchall()
            
            package = self.packages[license_row[2]]
            
            return {
                "license_key": license_key,
                "customer_id": license_row[1],
                "package": asdict(package),
                "issue_date": license_row[3],
                "expiry_date": license_row[4],
                "is_active": license_row[5],
                "usage_stats": json.loads(license_row[6]),
                "registered_devices": len(device_rows),
                "device_limit": package.device_limit,
                "devices_available": package.device_limit - len(device_rows)
            }
    
    def deactivate_device(self, license_key: str, device_id: str) -> bool:
        """Remove device from license (admin function)"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE registered_devices SET is_active = 0 WHERE license_key = ? AND device_id = ?', 
                          (license_key, device_id))
            affected_rows = cursor.rowcount
            conn.commit()
            
            if affected_rows > 0:
                logger.info(f"✅ Deactivated device {device_id} from license {license_key}")
                return True
            return False

# 🧪 **Demo Implementation**
def demo_adaptml_license_system():
    """Demonstrate AdaptML licensing system"""
    print("🔐 AdaptML Device Authentication & Licensing Demo")
    print("=" * 60)
    
    # Initialize license manager
    manager = AdaptMLLicenseManager("demo_adaptml_licenses.db")
    
    # Create sample licenses
    print("\n1. Creating sample licenses...")
    
    entry_license = manager.create_license("customer-001", "entry", is_lifetime=False)
    pro_lifetime = manager.create_license("customer-002", "pro", is_lifetime=True)
    max_license = manager.create_license("customer-003", "max", is_lifetime=False)
    
    print(f"✅ Entry License: {entry_license}")
    print(f"✅ Pro Lifetime: {pro_lifetime}")
    print(f"✅ Max License: {max_license}")
    
    # Test device registration
    print("\n2. Testing device registration...")
    
    success, message = manager.register_device(pro_lifetime)
    print(f"Registration result: {success} - {message}")
    
    # Test license validation
    print("\n3. Testing license validation...")
    
    valid, status, info = manager.validate_device_license(pro_lifetime)
    print(f"Validation result: {valid} - {status}")
    if valid:
        print(f"License info: {info}")
    
    # Test license information
    print("\n4. License information...")
    
    license_info = manager.get_license_info(pro_lifetime)
    if license_info:
        print(f"Package: {license_info['package']['package_name']}")
        print(f"Device limit: {license_info['device_limit']}")
        print(f"Devices registered: {license_info['registered_devices']}")
        print(f"Features: {', '.join(license_info['package']['features'])}")
    
    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    demo_adaptml_license_system()

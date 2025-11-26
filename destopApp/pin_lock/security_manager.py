import hashlib
import keyring
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Use a unique service name for your application
# This is how the credential will be identified in the OS keychain
SERVICE_NAME = "MCQ_Test_App"
USERNAME = "user_pin"

class SecurityManager:
    """Handles hashing, verification, and secure storage of the PIN."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecurityManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if SecurityManager._initialized:
            return
            
        # Argon2 is the current gold standard for password hashing
        self._ph = PasswordHasher(
            time_cost=3,      # Increases the number of iterations
            memory_cost=65536, # Uses 64 MB of RAM (65536 KiB)
            parallelism=4,    # Uses 4 threads
            hash_len=16,      # Length of the final hash
            salt_len=16       # Length of the random salt
        )
        self._aes_key = None  # Placeholder for AES key if needed later
        SecurityManager._initialized = True

    def is_pin_set(self) -> bool:
        """Check if a PIN is already stored in the OS keychain."""
        return keyring.get_password(SERVICE_NAME, USERNAME) is not None

    def set_pin(self, pin: str) -> None:
        """Hashes and stores a new PIN securely."""
        hashed_pin = self._ph.hash(pin)
        keyring.set_password(SERVICE_NAME, USERNAME, hashed_pin)

    def verify_pin(self, pin: str) -> bool:
        """Verifies an entered PIN against the stored hash."""
        try:
            stored_hash = keyring.get_password(SERVICE_NAME, USERNAME)
            if stored_hash is None:
                return False # No PIN is set
            
            # This will raise an exception if the PIN doesn't match
            self._ph.verify(stored_hash, pin)
            return True
        except VerifyMismatchError:
            # The PIN was incorrect
            return False
        except Exception as e:
            # Handle other potential errors, e.g., keyring access issues
            print(f"An unexpected security error occurred: {e}")
            return False
        
    def set_project_pw(self,pw:str):
        """Derives and stores the AES key from the project password (PIN)."""
        self._aes_key = self.derive_key_from_project_pw(pw)
    
    def derive_key_from_project_pw(self,pin: str) -> bytes:
        """
        Create a 32-byte key from the PIN using PBKDF2-HMAC-SHA256.
        """
        salt = b'' 
        iterations = 390000 
        key_length = 32
        hash_algorithm = 'sha256'
        
        key = hashlib.pbkdf2_hmac(
            hash_algorithm,
            pin.encode('utf-8'),
            salt, 
            iterations,
            dklen=key_length
        )
        return key
    def get_aes_key(self) -> bytes:
        """Returns the derived AES key after successful PIN verification."""
        return self._aes_key
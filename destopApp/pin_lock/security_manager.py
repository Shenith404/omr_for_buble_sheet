import keyring
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Use a unique service name for your application
# This is how the credential will be identified in the OS keychain
SERVICE_NAME = "MCQ_Test_App"
USERNAME = "user_pin"

class SecurityManager:
    """Handles hashing, verification, and secure storage of the PIN."""
    
    def __init__(self):
        # Argon2 is the current gold standard for password hashing
        self._ph = PasswordHasher(
            time_cost=3,      # Increases the number of iterations
            memory_cost=65536, # Uses 64 MB of RAM (65536 KiB)
            parallelism=4,    # Uses 4 threads
            hash_len=16,      # Length of the final hash
            salt_len=16       # Length of the random salt
        )

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
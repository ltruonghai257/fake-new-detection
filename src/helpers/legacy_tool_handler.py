import os
from typing import Callable, Any, Dict

class LegacyToolHandler:
    """
    A utility class to manage environment variables and configurations for older tools,
    especially concerning SSL/TLS compatibility.
    """

    def __init__(self, openssl_conf_path: str = "openssl.cnf"):
        self.openssl_conf_path = openssl_conf_path
        self._original_openssl_conf = os.environ.get("OPENSSL_CONF")

    def set_openssl_conf(self):
        """
        Sets the OPENSSL_CONF environment variable to the specified path.
        """
        if os.path.exists(self.openssl_conf_path):
            os.environ["OPENSSL_CONF"] = self.openssl_conf_path
            print(f"--- OPENSSL_CONF set to: {self.openssl_conf_path} ---")
        else:
            print(f"Warning: openssl.cnf not found at {self.openssl_conf_path}. OPENSSL_CONF not set.")

    def unset_openssl_conf(self):
        """
        Unsets the OPENSSL_CONF environment variable, restoring its original value if any.
        """
        if self._original_openssl_conf is not None:
            os.environ["OPENSSL_CONF"] = self._original_openssl_conf
            print(f"--- OPENSSL_CONF restored to original value: {self._original_openssl_conf} ---")
        else:
            del os.environ["OPENSSL_CONF"]
            print("--- OPENSSL_CONF unset. ---")

    def run_with_openssl_conf(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Executes a function with OPENSSL_CONF set, and then unsets it.
        """
        self.set_openssl_conf()
        try:
            return func(*args, **kwargs)
        finally:
            self.unset_openssl_conf()

    def __enter__(self):
        """Context manager entry: sets OPENSSL_CONF."""
        self.set_openssl_conf()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: unsets OPENSSL_CONF."""
        self.unset_openssl_conf()
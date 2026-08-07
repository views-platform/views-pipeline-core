"""auth.py — extracted from modules/appwrite/file.py (M-1 audit decision).

This module contains the classes that were previously in the
2,841-LOC God module `file.py`. The original `file.py` is now a
re-export shim that preserves all existing import paths.
"""

from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases
from appwrite.services.users import Users
from appwrite.input_file import InputFile
from appwrite.id import ID
from appwrite.exception import AppwriteException
from appwrite.query import Query
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import logging
from views_pipeline_core.modules.appwrite.config import AppwriteConfig, AuthMethod, OperationResult

logger = logging.getLogger(__name__)




class AuthManager(ABC):
    """Abstract base class for Appwrite authentication handlers.

    Defines the interface for authentication strategies used by AppWriteFileModule.
    Implementations must provide the setup() method to configure client authentication.

    See Also:
        ApiKeyAuth: Implementation for API key authentication.
        AuthFactory: Factory for creating appropriate AuthManager instances.
    """

    @abstractmethod
    def setup(self, client: Client, credentials: Union[str, Dict[str, str]]) -> OperationResult:
        """Configure authentication on the Appwrite client.

        Args:
            client: Appwrite Client instance to configure.
            credentials: Authentication credentials (format depends on implementation).

        Returns:
            OperationResult indicating success or failure of authentication setup.
        """
        pass


class ApiKeyAuth(AuthManager):
    """API key authentication handler for server-side Appwrite access.

    Uses a server API key for authentication, suitable for backend services
    and automated pipelines. Provides full access based on API key permissions.

    Example:
        >>> auth = ApiKeyAuth()
        >>> result = auth.setup(client, "my_api_key_string")
        >>> if result.success:
        ...     # Client is now authenticated
    """

    def setup(self, client: Client, credentials: Union[str, Dict[str, str]]) -> OperationResult:
        """Configure API key authentication on the client.

        Args:
            client: Appwrite Client instance to configure.
            credentials: API key string. Must be a string, not a dictionary.

        Returns:
            OperationResult with success=True if key was set, or success=False
            with error message if credentials format is invalid.
        """
        if not isinstance(credentials, str):
            return OperationResult(
                success=False,
                error="API key authentication requires string credentials",
                code="INVALID_CREDENTIALS"
            )
        
        client.set_key(credentials)
        return OperationResult(success=True)


class AuthFactory:
    """Factory for creating authentication handler instances.

    Provides a static method to instantiate the appropriate AuthManager
    subclass based on the specified authentication method.

    Example:
        >>> auth_manager = AuthFactory.create_auth(AuthMethod.API_KEY)
        >>> result = auth_manager.setup(client, "my_api_key")
    """

    @staticmethod
    def create_auth(auth_method: AuthMethod) -> AuthManager:
        """Create an AuthManager instance for the specified method.

        Args:
            auth_method: The authentication method to use.

        Returns:
            Appropriate AuthManager subclass instance.

        Raises:
            ValueError: If auth_method is not supported.
        """
        if auth_method == AuthMethod.API_KEY:
            return ApiKeyAuth()
        raise ValueError(f"Unsupported auth method: {auth_method}")

# Cache Management
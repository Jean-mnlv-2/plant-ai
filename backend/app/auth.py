"""
Utilitaires d'authentification pour l'API Plant-AI.
"""
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import hmac
import json
import os
import time

import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .models import User, UserRole, Tokens, UserPreferences
from .database import db
from .settings import settings

# Configuration JWT (résolue à l'usage pour éviter l'échec au démarrage)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))

security = HTTPBearer()


def _b64url(data: bytes) -> str:
    """Encode bytes to base64url string."""
    return data.replace(b"+", b"-").replace(b"/", b"_").replace(b"=", b"").decode()


def _sign(data: bytes, secret: str) -> str:
    """Sign data with HMAC-SHA256."""
    return _b64url(hmac.new(secret.encode(), data, hashlib.sha256).digest())


def _load_jwt_secret_from_file() -> Optional[str]:
    try:
        if settings.jwt_secret_file and settings.jwt_secret_file.exists():
            return settings.jwt_secret_file.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return None


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if secret:
        return secret
    file_secret = _load_jwt_secret_from_file()
    if file_secret:
        return file_secret
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="JWT secret not configured. Set JWT_SECRET or provide models/.jwt_secret",
    )


def jwt_encode(payload: Dict[str, Any], secret: Optional[str] = None) -> str:
    """Encode JWT token."""
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    actual_secret = secret or get_jwt_secret()
    return jwt.encode(payload, actual_secret, algorithm=JWT_ALGORITHM)


def jwt_decode(token: str, secret: Optional[str] = None) -> Dict[str, Any]:
    """Decode JWT token."""
    try:
        actual_secret = secret or get_jwt_secret()
        return jwt.decode(token, actual_secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = os.urandom(32)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return salt.hex() + pwdhash.hex()


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    try:
        salt = bytes.fromhex(hashed[:64])
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return pwdhash.hex() == hashed[64:]
    except (ValueError, IndexError):
        return False


def generate_tokens(user_id: str, user_role: UserRole) -> Tokens:
    """Generate access and refresh tokens."""
    # Access token
    access_payload = {
        "sub": user_id,
        "role": user_role.value,
        "type": "access"
    }
    access_token = jwt_encode(access_payload)
    
    # Refresh token
    refresh_payload = {
        "sub": user_id,
        "type": "refresh"
    }
    refresh_token = jwt_encode(refresh_payload)
    
    return Tokens(
        accessToken=access_token,
        refreshToken=refresh_token,
        expiresIn=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current user ID from JWT token."""
    token = credentials.credentials
    payload = jwt_decode(token)
    
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type"
        )
    
    return payload.get("sub")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token."""
    user_id = get_current_user_id(credentials)
    
    # Get user from database
    user_data = db.get_user_by_id(user_id)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return User(**user_data)


def require_role(required_role: UserRole):
    """Decorator to require specific user role."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != required_role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role.value} role"
            )
        return current_user
    return role_checker


def create_user(name: str, email: str, password: str, country: str, role: UserRole = UserRole.FARMER) -> User:
    """Create a new user."""
    # Check if user already exists
    if db.get_user_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Generate user ID
    user_id = f"user_{int(time.time() * 1000)}"
    
    # Hash password
    hashed_password = hash_password(password)
    
    # Create user
    user = User(
        id=user_id,
        name=name,
        email=email,
        country=country,
        role=role,
        createdAt=datetime.utcnow(),
        preferences=UserPreferences().dict()
    )
    
    # Save to database
    db.create_user(user.dict(), hashed_password)
    
    return user


def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    user_data = db.get_user_by_email(email)
    if not user_data:
        return None
    
    stored_password = db.get_user_password(email)
    if not stored_password or not verify_password(password, stored_password):
        return None
    
    return User(**user_data)

"""
User Domain Entity & Data Model
-------------------------------
Typed dataclass representing user account structure in MongoDB.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.models.user_settings import UserSettings

@dataclass
class User:
    username: str
    email: str
    password: str
    roles: List[str] = field(default_factory=lambda: ["user"])
    settings: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        if self.settings is None:
            self.settings = UserSettings(full_name=self.username).to_dict()
        self.email = self.email.strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "username": self.username,
            "email": self.email,
            "password": self.password,
            "roles": self.roles,
            "settings": self.settings,
            "created_at": self.created_at
        }
        if self.id:
            data["_id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        if not data:
            return None
        return cls(
            username=data.get("username", "User"),
            email=data.get("email", ""),
            password=data.get("password", ""),
            roles=data.get("roles", ["user"]),
            settings=data.get("settings"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            id=str(data["_id"]) if "_id" in data else None
        )

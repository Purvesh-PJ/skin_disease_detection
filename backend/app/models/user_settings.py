"""
User Settings Domain Entity & Data Model
----------------------------------------
Typed dataclass representing user preferences and clinical profile metadata.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class UserSettings:
    full_name: str
    role_title: str = "Clinical Evaluator"
    specialization: str = "Dermatology AI"
    theme: str = "dark"
    email_notifications: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSettings":
        if not data:
            return None
        return cls(
            full_name=data.get("full_name", "User"),
            role_title=data.get("role_title", "Clinical Evaluator"),
            specialization=data.get("specialization", "Dermatology AI"),
            theme=data.get("theme", "dark"),
            email_notifications=data.get("email_notifications", True)
        )

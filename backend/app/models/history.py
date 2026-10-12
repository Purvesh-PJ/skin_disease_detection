"""
Prediction History Domain Entity & Data Model
---------------------------------------------
Typed dataclass representing scan history record stored in MongoDB.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class PredictionHistory:
    user_email: str
    predicted_disease: str
    confidence: str
    filename: str = "unknown_lesion.jpg"
    disease_details: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    id: Optional[str] = None

    def __post_init__(self):
        self.user_email = self.user_email.strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "user_email": self.user_email,
            "predicted_disease": self.predicted_disease,
            "confidence": self.confidence,
            "disease_details": self.disease_details,
            "filename": self.filename,
            "created_at": self.created_at
        }
        if self.id:
            data["_id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionHistory":
        if not data:
            return None
        return cls(
            user_email=data.get("user_email", ""),
            predicted_disease=data.get("predicted_disease", ""),
            confidence=data.get("confidence", "0"),
            filename=data.get("filename", "unknown_lesion.jpg"),
            disease_details=data.get("disease_details", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            id=str(data["_id"]) if "_id" in data else None
        )

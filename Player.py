from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class Player: #these will determine the starting stats of the player
    # Personal Info & Calendar Tracker
    name: str
    age: int
    month: str
    year: int
    location: str

    #Core stats
    health: int
    strength: int
    charisma: int
    intelligence: int
    willpower: int
    stress: int

    cash: float # Liquid spending money ($)
    # Character context
    occupation: str = ""
    background: str = ""

    # Persistent simulation state
    skills: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    relationships: Dict[str, int] = field(default_factory=dict)
    reputation: Dict[str, int] = field(default_factory=dict)
    debt: float = 0.0
    status_flags: List[str] = field(default_factory=list)

    def export_engine_state(self) -> Dict[str, Any]:
        """Packages the player state into a dictionary for internal tracking or LLM payloads."""
        return {
            "header": f"{self.name} | Age: {self.age} | Month: {self.month} Year: {self.year} | Location: {self.location}",
            "stats": {
                "health": self.health,
                "strength": self.strength,
                "charisma": self.charisma,
                "intelligence": self.intelligence,
                "willpower": self.willpower,
                "stress": self.stress,
            },
            "cash": self.cash,
            "occupation": self.occupation,
            "background": self.background,
            "skills": self.skills,
            "inventory": self.inventory,
            "relationships": self.relationships,
            "reputation": self.reputation,
            "debt": self.debt,
            "status_flags": self.status_flags,
        }
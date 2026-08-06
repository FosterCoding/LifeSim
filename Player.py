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


    def apply_deltas(self, state_deltas: Dict[str, Any]) -> None:
        """
        Applies state updates returned by narrate_outcome to the player object.
        Supports stat increments/decrements, list additions/removals, and dict updates.
        """
        if not isinstance(state_deltas, dict):
            return
        for key, delta in state_deltas.items():
            # 1. Update Numeric Attributes
            if hasattr(self, key):
                current_val = getattr(self, key)                
                # Direct numeric adjustments (e.g. "health": -2, "cash": 50.0, "stress": 3)
                if isinstance(current_val, (int, float)) and isinstance(delta, (int, float)):
                    new_val = current_val + delta                
                    # Clamp Core D&D Stats & Health between 0 and 20
                    if key in {"health", "strength", "charisma", "intelligence", "willpower"}:
                        new_val = max(0, min(20, int(new_val)))
                    # Clamp Stress between 0 and 20
                    elif key == "stress":
                        new_val = max(0, min(20, int(new_val)))
                    setattr(self, key, new_val)
                # 2. Append to Lists (e.g. "inventory": ["Lockpick"], "status_flags": ["Wounded"])
                elif isinstance(current_val, list) and isinstance(delta, list):
                    for item in delta:
                        if item not in current_val:
                            current_val.append(item)
                # 3. Update Dictionary Mappings (e.g. "skills", "relationships", "reputation")
                elif isinstance(current_val, dict) and isinstance(delta, dict):
                    current_val.update(delta)
            # 4. Handle Item Removals (e.g. "remove_inventory": ["Crowbar"])
            if key == "remove_inventory" and isinstance(delta, list):
                for item in delta:
                    if item in self.inventory:
                        self.inventory.remove(item)
            # 5. Handle Status Flag Removals (e.g. "remove_status": ["Wounded"])
            if key == "remove_status" and isinstance(delta, list):
                for item in delta:
                    if item in self.status_flags:
                        self.status_flags.remove(item)

    
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

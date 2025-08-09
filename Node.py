import json
import uuid
from typing import List, Optional, Dict, Any
import networkx as nx


class Entity:
    """Represents a research entity, such as a field, topic, or keyword."""
    def __init__(self, name: str, field: Optional[str] = None, description: Optional[str] = None):
        """
        Initializes an Entity object.

        Args:
            name: The name of the entity (e.g., "Machine Learning").
            field: The field of the entity (e.g., "Computer Science").
            description: A brief description of the entity (e.g., "Study of algorithms that improve automatically through experience").
        """
        self._id = uuid.uuid4()  # Unique identifier for the entity
        self.name = name
        self.field = field
        self.description = description

    def __eq__(self, other):
        return isinstance(other, Entity) and self._id == other._id
    
    def __hash__(self):
        return hash(self._id) # 这里选择哈希id, 因为可能会创建name一样的Entity, 比如最经典的LLM(Large Language Model)和LLM(Legum Magister)所以不得不使用uuid
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the Entity object to a dictionary."""
        return {"id": self._id, "name": self.name, "field": self.field, "description": self.description}

    def to_json(self, indent: int = 4) -> str:
        """Converts the Entity object to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return f"Entity(name='{self.name}'\nfield='{self.field}'\ndescription='{self.description}')"

class Paper:
    """Represents a single research paper."""
    def __init__(self, title: str, abstract: str ,field: Optional[str] = None):
        """
        Initializes a Paper object.

        Args:
            abstract: The abstract of the paper.
        """
        self.Title = title
        self.field = field 
        self.Abstract = abstract

    def __eq__(self, other):
        return isinstance(other, Paper) and self.Title == other.Title
    
    def __hash__(self):
        return hash(self.Title) # paper的标题是唯一的, 所以可以直接使用标题作为哈希值

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Paper object to a dictionary.
        To avoid circular dependencies, authors are represented by their names.
        """
        return {
            "Title": self.Title,
            "Abstract": self.Abstract,
            "Field": self.field
        }

    def to_json(self, indent: int = 4) -> str:
        """Converts the Paper object to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return f"Paper(Title='{self.Title}'\nfield='{self.field}')"

class Author:
    """Represents a single author."""
    def __init__(self, name: str, email: Optional[str] = None):
        """
        Initializes an Author object.

        Args:
            name: The name of the author.
            email: The author's email address (optional).
        """
        self._id = uuid.uuid4()  # Unique identifier for the author
        self.Name: str = name
        self.Email: Optional[str] = email

    def __eq__(self, other):
        return isinstance(other, Author) and self._id == other._id
    
    def __hash__(self):
        return hash(self._id)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Author object to a dictionary.
        To avoid circular dependencies, papers are represented by their abstracts.
        """
        return {
            "Name": self.Name,
            "Email": self.Email,
        }

    def to_json(self, indent: int = 4) -> str:
        """Converts the Author object to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return f"Author(Name: {self.Name}\nEmail: {self.Email or 'N/A'})"


class Affilation:
    """Represents an affiliation, such as a university or research institution."""
    def __init__(self, name: str):
        """
        Initializes an Affiliation object.

        Args:
            name: The name of the affiliation (e.g., "MIT").
        """
        self.Name: str = name

    def __eq__(self, other):
        return isinstance(other, Affilation) and self.Name == other.Name
    
    def __hash__(self):
        return hash(self.Name)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Affiliation object to a dictionary."""
        return {"Name": self.Name}

    def to_json(self, indent: int = 4) -> str:
        """Converts the Affiliation object to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return f"Affiliation(Name='{self.Name}')"
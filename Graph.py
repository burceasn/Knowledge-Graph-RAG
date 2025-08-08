import json
from typing import List, Optional, Dict, Any
import networkx as nx


class Entity:
    """Represents a research entity, such as a field, topic, or keyword."""
    def __init__(self, name: str, entity_type: Optional[str] = None, description: Optional[str] = None):
        """
        Initializes an Entity object.

        Args:
            name: The name of the entity (e.g., "Machine Learning").
            entity_type: The type of the entity (e.g., "Field").
        """
        self.name = name
        self.entity_type = entity_type
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Entity object to a dictionary."""
        return {"name": self.name, "entity_type": self.entity_type}

    def to_json(self, indent: int = 4) -> str:
        """Converts the Entity object to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return f"Entity(name='{self.name}', entity_type='{self.entity_type}')"

class Paper:
    """Represents a single research paper."""
    def __init__(self, title: str, abstract: str):
        """
        Initializes a Paper object.

        Args:
            abstract: The abstract of the paper.
        """
        self.Title: str = title
        self.Authors: List['Author'] = []
        self.Abstract: str = abstract
        self.Research_Entity: List['Entity'] = []

    def add_author(self, author: 'Author'):
        """Adds an author to the paper's author list."""
        if author not in self.Authors:
            self.Authors.append(author)

    def add_research_entity(self, entity: 'Entity'):
        """Adds a research entity related to the paper."""
        if entity not in self.Research_Entity:
            self.Research_Entity.append(entity)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Paper object to a dictionary.
        To avoid circular dependencies, authors are represented by their names.
        """
        return {
            "Title": self.Title,
            "Authors": [author.Name for author in self.Authors],
            "Abstract": self.Abstract,
            "Research_Entity": [entity.to_dict() for entity in self.Research_Entity]
        }

    def to_json(self, indent: int = 4) -> str:
        """Converts the Paper object to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        co_authors = ", ".join(author.Name for author in self.Authors)
        return f"Paper(Title='{self.Title}', Authors=[{co_authors}])"

class Author:
    """Represents a single author."""
    def __init__(self, name: str, affiliation: str, email: Optional[str] = None):
        """
        Initializes an Author object.

        Args:
            name: The name of the author.
            affiliation: The author's affiliation (e.g., university or company).
            email: The author's email address (optional).
        """
        self.Name: str = name
        self.Email: Optional[str] = email
        self.Affiliation: str = affiliation
        self.Papers: List['Paper'] = []
        self.Research_Entity: List['Entity'] = []

    def add_paper(self, paper: 'Paper'):
        """
        Adds a paper to the author's list of papers and ensures the
        author is also added to the paper's author list.
        """
        if paper not in self.Papers:
            self.Papers.append(paper)
            paper.add_author(self)

    def add_research_entity(self, entity: 'Entity'):
        """Adds a research entity (e.g., an interest) to the author."""
        if entity not in self.Research_Entity:
            self.Research_Entity.append(entity)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Author object to a dictionary.
        To avoid circular dependencies, papers are represented by their abstracts.
        """
        return {
            "Name": self.Name,
            "Email": self.Email,
            "Affiliation": self.Affiliation,
            "Papers": [paper.Abstract for paper in self.Papers],
            "Research_Entity": [entity.to_dict() for entity in self.Research_Entity]
        }

    def to_json(self, indent: int = 4) -> str:
        """Converts the Author object to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        papers = "\n".join([f"  - {paper.Title}" for paper in self.Papers])
        return f"Author(\n  Name: {self.Name}\n  Affiliation: {self.Affiliation}\n  Email: {self.Email or 'N/A'}\n  Papers:\n{papers}\n)"


class Affilation:
    """Represents an affiliation, such as a university or research institution."""
    def __init__(self, name: str):
        """
        Initializes an Affiliation object.

        Args:
            name: The name of the affiliation (e.g., "MIT").
        """
        self.Name: str = name
        self.Authors: List[Author] = []

    def add_author(self, author: Author):
        """Adds an author to the affiliation's author list."""
        if author not in self.Authors:
            self.Authors.append(author)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Affiliation object to a dictionary."""
        return {
            "Name": self.Name,
            "Authors": [author.Name for author in self.Authors]
        }

    def to_json(self, indent: int = 4) -> str:
        """Converts the Affiliation object to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        return f"Affiliation(Name='{self.Name}')"
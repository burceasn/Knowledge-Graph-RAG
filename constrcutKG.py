import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

# Import your existing classes
from Author_metadata import AuthorList, Author as AuthorMeta
from Node import Author, Paper, Affiliation, Entity
from Edge import (
    AuthorPaperEdge, 
    AuthorAffiliationEdge, 
    AuthorCoauthorEdge,
    PaperAffiliationEdge,
    AffiliationCollaborationEdge
)

class KnowledgeGraphBuilder:
    """
    A class to build and manage a knowledge graph using NetworkX.
    """
    
    def __init__(self):
        """Initialize the knowledge graph."""
        self.graph = nx.MultiDiGraph()  # Using MultiDiGraph to allow multiple edges between nodes
        self.authors: Dict[str, Author] = {}  # Cache for author nodes by name
        self.affiliations: Dict[str, Affiliation] = {}  # Cache for affiliation nodes by name
        self.papers: Dict[str, Paper] = {}  # Cache for paper nodes by title
        self.edges: List = []  # Store all edge objects
        
    def get_or_create_author(self, name: str, email: Optional[str] = None) -> Author:
        """
        Get existing author or create new one.
        
        Args:
            name: Author's name
            email: Author's email (optional)
            
        Returns:
            Author node object
        """
        if name not in self.authors:
            author = Author(name=name, email=email)
            self.authors[name] = author
            self.graph.add_node(author._id, 
                               node_type='Author',
                               name=name, 
                               email=email,
                               node_object=author)
        else:
            # Update email if provided and not already set
            if email and not self.authors[name].Email:
                self.authors[name].Email = email
                self.graph.nodes[self.authors[name]._id]['email'] = email
        
        return self.authors[name]
    
    def get_or_create_affiliation(self, name: str) -> Affiliation:
        """
        Get existing affiliation or create new one.
        
        Args:
            name: Affiliation's name
            
        Returns:
            Affiliation node object
        """
        if name not in self.affiliations:
            affiliation = Affiliation(name=name)
            self.affiliations[name] = affiliation
            self.graph.add_node(name,  # Using name as ID for affiliations
                               node_type='Affiliation',
                               name=name,
                               node_object=affiliation)
        
        return self.affiliations[name]
    
    def add_paper(self, paper: Paper) -> Paper:
        """
        Add a paper to the graph.
        
        Args:
            paper: Paper object
            
        Returns:
            The paper object
        """
        if paper.Title not in self.papers:
            self.papers[paper.Title] = paper
            self.graph.add_node(paper.Title,
                               node_type='Paper',
                               title=paper.Title,
                               abstract=paper.Abstract,
                               field=paper.field,
                               node_object=paper)
        
        return self.papers[paper.Title]
    
    def process_author_list_and_paper(self, author_list: AuthorList, paper: Paper):
        """
        Process an AuthorList and Paper to create nodes and edges.
        
        Args:
            author_list: AuthorList object containing authors
            paper: Paper object
        """
        # Add paper to graph
        paper_node = self.add_paper(paper)
        
        # Track affiliations for this paper
        paper_affiliations = set()
        
        # Process each author
        author_nodes = []
        for author_meta in author_list.Authors:
            # Create author node
            author_node = self.get_or_create_author(
                name=author_meta.Name,
                email=author_meta.Email
            )
            author_nodes.append(author_node)
            
            # Create Author-Paper edge
            author_order = author_meta.Author_Order if hasattr(author_meta, 'Author_Order') and author_meta.Author_Order else 0
            author_paper_edge = AuthorPaperEdge(
                author=author_node,
                paper=paper_node,
                author_order=author_order
            )
            self.edges.append(author_paper_edge)
            self.graph.add_edge(
                author_node._id,
                paper.Title,
                relation=author_paper_edge.relation,
                edge_type='AuthorPaper',
                author_order=author_order,
                weight=0.0,
                edge_object=author_paper_edge
            )
            
            # Process affiliation if exists
            if hasattr(author_meta, 'Affiliation') and author_meta.Affiliation:
                affiliation_node = self.get_or_create_affiliation(author_meta.Affiliation)
                paper_affiliations.add(affiliation_node)
                
                # Create Author-Affiliation edge
                author_affiliation_edge = AuthorAffiliationEdge(
                    author=author_node,
                    affiliation=affiliation_node
                )
                self.edges.append(author_affiliation_edge)
                
                # Check if edge already exists before adding
                if not self.graph.has_edge(author_node._id, affiliation_node.Name):
                    self.graph.add_edge(
                        author_node._id,
                        affiliation_node.Name,
                        relation=author_affiliation_edge.relation,
                        edge_type='AuthorAffiliation',
                        rank=0,
                        edge_object=author_affiliation_edge
                    )
        
        # Create co-author edges
        for i in range(len(author_nodes)):
            for j in range(i + 1, len(author_nodes)):
                coauthor_edge = AuthorCoauthorEdge(
                    author1=author_nodes[i],
                    author2=author_nodes[j],
                    coauthored_paper=paper_node
                )
                self.edges.append(coauthor_edge)
                
                # Check if coauthor edge already exists
                existing_edge = None
                if self.graph.has_edge(coauthor_edge.source._id, coauthor_edge.target._id):
                    for key, edge_data in self.graph[coauthor_edge.source._id][coauthor_edge.target._id].items():
                        if edge_data.get('edge_type') == 'Coauthor':
                            existing_edge = edge_data.get('edge_object')
                            break
                
                if existing_edge:
                    # Update existing edge with new paper
                    existing_edge.add_coauthored_paper(paper_node)
                else:
                    # Add new edge
                    self.graph.add_edge(
                        coauthor_edge.source._id,
                        coauthor_edge.target._id,
                        relation=coauthor_edge.relation,
                        edge_type='Coauthor',
                        weight=0.0,
                        coauthored_papers=[paper_node],
                        edge_object=coauthor_edge
                    )
        
        # Create Paper-Affiliation edges
        for affiliation_node in paper_affiliations:
            paper_affiliation_edge = PaperAffiliationEdge(
                paper=paper_node,
                affiliation=affiliation_node
            )
            self.edges.append(paper_affiliation_edge)
            self.graph.add_edge(
                paper.Title,
                affiliation_node.Name,
                relation=paper_affiliation_edge.relation,
                edge_type='PaperAffiliation',
                edge_object=paper_affiliation_edge
            )
        
        # Create Affiliation collaboration edges
        affiliation_list = list(paper_affiliations)
        for i in range(len(affiliation_list)):
            for j in range(i + 1, len(affiliation_list)):
                collab_edge = AffiliationCollaborationEdge(
                    affiliation1=affiliation_list[i],
                    affiliation2=affiliation_list[j],
                    collaboration_paper=paper_node
                )
                self.edges.append(collab_edge)
                
                # Check if collaboration edge already exists
                existing_edge = None
                if self.graph.has_edge(collab_edge.source.Name, collab_edge.target.Name):
                    for key, edge_data in self.graph[collab_edge.source.Name][collab_edge.target.Name].items():
                        if edge_data.get('edge_type') == 'AffiliationCollaboration':
                            existing_edge = edge_data.get('edge_object')
                            break
                
                if existing_edge:
                    # Update existing edge with new paper
                    existing_edge.add_collaboration_paper(paper_node)
                else:
                    # Add new edge
                    self.graph.add_edge(
                        collab_edge.source.Name,
                        collab_edge.target.Name,
                        relation=collab_edge.relation,
                        edge_type='AffiliationCollaboration',
                        weight=0.0,
                        collaboration_papers=[paper_node],
                        edge_object=collab_edge
                    )
    
    def get_graph_statistics(self) -> Dict:
        """
        Get basic statistics about the graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for node, data in self.graph.nodes(data=True):
            node_types[data.get('node_type', 'Unknown')] += 1
        
        for u, v, data in self.graph.edges(data=True):
            edge_types[data.get('edge_type', 'Unknown')] += 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'is_connected': nx.is_weakly_connected(self.graph),
            'number_of_components': nx.number_weakly_connected_components(self.graph)
        }
    
    
    def export_to_graphml(self, filename: str):
        """
        Export the graph to GraphML format.
        
        Args:
            filename: Output filename (should end with .graphml)
        """
        # Create a copy of the graph for export
        export_graph = self.graph.copy()
        
        # Convert node objects to serializable format
        for node, data in export_graph.nodes(data=True):
            if 'node_object' in data:
                del data['node_object']
            # Convert UUID to string if it's used as node ID
            if hasattr(node, '__class__') and 'UUID' in str(type(node)):
                data['uuid'] = str(node)
        
        # Convert edge objects to serializable format
        # In NetworkX 3.x, we need to iterate over edges differently for MultiDiGraph
        for u, v, data in export_graph.edges(data=True):
            if 'edge_object' in data:
                del data['edge_object']
            if 'coauthored_papers' in data:
                data['num_coauthored_papers'] = len(data['coauthored_papers'])
                del data['coauthored_papers']
            if 'collaboration_papers' in data:
                data['num_collaboration_papers'] = len(data['collaboration_papers'])
                del data['collaboration_papers']
            # Convert any Paper objects in coauthored_paper_list to titles
            if 'coauthored_paper_list' in data:
                data['coauthored_papers'] = [p.Title for p in data['coauthored_paper_list']]
                del data['coauthored_paper_list']
            if 'collaboration_paper_list' in data:
                data['collaboration_papers'] = [p.Title for p in data['collaboration_paper_list']]
                del data['collaboration_paper_list']
        
        # Ensure filename ends with .graphml
        if not filename.endswith('.graphml'):
            filename += '.graphml'
        
        nx.write_graphml(export_graph, filename)
        print(f"Graph exported to {filename}")
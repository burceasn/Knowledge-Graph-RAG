import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import json
from uuid import UUID

# Custom JSON encoder for UUID serialization
class UUIDEncoder(json.JSONEncoder):
    """
    A custom JSON encoder to handle UUID objects, converting them to strings.
    """
    def default(self, o):
        if isinstance(o, UUID):
            return str(o)
        return super().default(o)


# Import your existing classes
from Author_metadata import AuthorList, Author as AuthorMeta
from Node import Author, Paper, Affiliation, Entity
from Edge import (
    AuthorPaperEdge, 
    AuthorAffiliationEdge, 
    AuthorCoauthorEdge,
    PaperAffiliationEdge,
    AffiliationCollaborationEdge,
    PaperCitationEdge,
    PaperEntityEdge
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
        self.entities: Dict[str, Entity] = {}  # Cache for entity nodes by name
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
                               display_name=name,  # 添加显示名称
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
                               display_name=name,  # 添加显示名称
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
                               display_name=paper.Title,  # 添加显示名称
                               node_object=paper)
        
        return self.papers[paper.Title]
    
    def get_or_create_entity(self, name: str, field: Optional[str] = None, 
                            description: Optional[str] = None) -> Entity:
        """
        Get existing entity or create new one.
        
        Args:
            name: Entity's name
            field: Entity's field (optional)
            description: Entity's description (optional)
            
        Returns:
            Entity node object
        """
        if name not in self.entities:
            entity = Entity(name=name, field=field, description=description)
            self.entities[name] = entity
            self.graph.add_node(entity._id,
                               node_type='Entity',
                               name=name,
                               field=field,
                               description=description,
                               display_name=name,  # 添加显示名称
                               node_object=entity)
        
        return self.entities[name]
    
    def find_existing_edge(self, source_id, target_id, edge_type: str):
        """
        辅助函数：查找特定类型的边
        
        Returns:
            tuple: (edge_object, edge_key) 如果找到，否则 (None, None)
        """
        if not self.graph.has_edge(source_id, target_id):
            return None, None
        
        edges_dict = self.graph[source_id][target_id]
        for key, data in edges_dict.items():
            if data.get('edge_type') == edge_type:
                return data.get('edge_object'), key
        
        return None, None

    def update_edge_in_graph(self, source_id, target_id, edge_key, updates: dict):
        """
        辅助函数：更新图中的边数据
        """
        if self.graph.has_edge(source_id, target_id) and edge_key in self.graph[source_id][target_id]:
            for key, value in updates.items():
                self.graph[source_id][target_id][edge_key][key] = value

    def calculate_author_credit(self, author_list: AuthorList, target_paper: Paper) -> Dict[str, float]:
        """
        计算论文中每个作者的权重（使用 Harmonic Credit Model）
        
        Args:
            author_list: 作者列表
            target_paper: 目标论文
            
        Returns:
            Dict[str, float]: 作者名称到权重的映射
        """
        # 获取所有作者的顺序列表
        author_orders = []
        for author in author_list.Authors:
            # 如果没有 Author_Order，使用默认值 1
            order = getattr(author, 'Author_Order', 1) or 1
            author_orders.append(order)
        
        # 计算 Harmonic Credit Model
        # credit_i = (1/i) / sum(1/k for k in all_orders)
        sum_harmonic = sum(1/order for order in author_orders)
        
        # 创建作者权重映射
        author_weights = {}
        for i, author in enumerate(author_list.Authors):
            order = author_orders[i]
            weight = round((1/order) / sum_harmonic, 4)  # 保留4位小数
            author_weights[author.Name] = weight
        
        return author_weights

    def process_author_list_and_paper(self, author_list: AuthorList, paper: Paper):
        """
        处理作者列表和论文，创建相关的节点和边
        这是修复了edge_data未绑定问题的版本
        """
        # Add paper to graph
        paper_node = self.add_paper(paper)
        
        # Track affiliations for this paper
        paper_affiliations = set()
        
        author_weights = self.calculate_author_credit(author_list, paper_node)

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
            author_order = getattr(author_meta, 'Author_Order', 1) or 1
            author_paper_edge = AuthorPaperEdge(
                author=author_node,
                paper=paper_node,
                author_order=author_order
            )

            # 更新权重
            weight = author_weights.get(author_meta.Name, 0.0)
            author_paper_edge.update_weight(weight)
            self.edges.append(author_paper_edge)
            
            self.graph.add_edge(
                author_node._id,
                paper.Title,
                relation=author_paper_edge.relation,
                edge_type='AuthorPaper',
                author_order=author_order,
                weight=0.0,
                edge_object=author_paper_edge,
                display_info=author_paper_edge.get_simple_display()
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
                        edge_object=author_affiliation_edge,
                        display_info=author_affiliation_edge.get_simple_display()
                    )
        
        # Create co-author edges (使用修复的版本)
        for i in range(len(author_nodes)):
            for j in range(i + 1, len(author_nodes)):
                coauthor_edge = AuthorCoauthorEdge(
                    author1=author_nodes[i],
                    author2=author_nodes[j],
                    coauthored_paper=paper_node
                )
                self.edges.append(coauthor_edge)
                
                source_id = coauthor_edge.source._id
                target_id = coauthor_edge.target._id
                
                # 查找现有边
                existing_edge, edge_key = self.find_existing_edge(source_id, target_id, 'Coauthor')
                
                if existing_edge:
                    # 更新现有边
                    existing_edge.add_coauthored_paper(paper_node)
                    # 更新图中的数据
                    self.update_edge_in_graph(source_id, target_id, edge_key, {
                        'display_info': existing_edge.get_simple_display(),
                        'coauthored_papers': existing_edge.attributes.get('coauthored_paper_list', [])
                    })
                else:
                    # 创建新边
                    self.graph.add_edge(
                        source_id,
                        target_id,
                        relation=coauthor_edge.relation,
                        edge_type='Coauthor',
                        weight=0.0,
                        coauthored_papers=[paper_node],
                        edge_object=coauthor_edge,
                        display_info=coauthor_edge.get_simple_display()
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
                edge_object=paper_affiliation_edge,
                display_info=paper_affiliation_edge.get_simple_display()
            )
        
        # Create Affiliation collaboration edges (同样需要修复)
        affiliation_list = list(paper_affiliations)
        for i in range(len(affiliation_list)):
            for j in range(i + 1, len(affiliation_list)):
                collab_edge = AffiliationCollaborationEdge(
                    affiliation1=affiliation_list[i],
                    affiliation2=affiliation_list[j],
                    collaboration_paper=paper_node
                )
                self.edges.append(collab_edge)
                
                source_name = collab_edge.source.Name
                target_name = collab_edge.target.Name
                
                # 查找现有边
                existing_edge, edge_key = self.find_existing_edge(source_name, target_name, 'AffiliationCollaboration')
                
                if existing_edge:
                    # 更新现有边
                    existing_edge.add_collaboration_paper(paper_node)
                    # 更新图中的数据
                    self.update_edge_in_graph(source_name, target_name, edge_key, {
                        'display_info': existing_edge.get_simple_display(),
                        'collaboration_papers': existing_edge.attributes.get('collaboration_paper_list', [])
                    })
                else:
                    # 创建新边
                    self.graph.add_edge(
                        source_name,
                        target_name,
                        relation=collab_edge.relation,
                        edge_type='AffiliationCollaboration',
                        weight=0.0,
                        collaboration_papers=[paper_node],
                        edge_object=collab_edge,
                        display_info=collab_edge.get_simple_display()
                    )
    
    def add_paper_entity_relation(self, paper: Paper, entity: Entity, weight: float = 0.0):
        """
        Add a relation between a paper and an entity.
        
        Args:
            paper: Paper object
            entity: Entity object
            weight: Initial weight of the relation
        """
        paper_node = self.add_paper(paper)
        entity_node = self.get_or_create_entity(
            name=entity.name,
            field=entity.field,
            description=entity.description
        )
        
        paper_entity_edge = PaperEntityEdge(
            paper=paper_node,
            entity=entity_node
        )
        paper_entity_edge.update_weight(weight)
        
        self.edges.append(paper_entity_edge)
        self.graph.add_edge(
            paper.Title,
            entity_node._id,
            relation=paper_entity_edge.relation,
            edge_type='PaperEntity',
            weight=weight,
            edge_object=paper_entity_edge,
            display_info=paper_entity_edge.get_simple_display()
        )
    
    def add_paper_citation(self, citing_paper: Paper, cited_paper: Paper):
        """
        Add a citation relation between papers.
        
        Args:
            citing_paper: The paper that cites
            cited_paper: The paper being cited
        """
        citing_node = self.add_paper(citing_paper)
        cited_node = self.add_paper(cited_paper)
        
        citation_edge = PaperCitationEdge(
            citing_paper=citing_node,
            cited_paper=cited_node
        )
        
        self.edges.append(citation_edge)
        self.graph.add_edge(
            citing_paper.Title,
            cited_paper.Title,
            relation=citation_edge.relation,
            edge_type='Citation',
            edge_object=citation_edge,
            display_info=citation_edge.get_simple_display()
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
    
    def display_edges(self, limit: int = 10, edge_type: Optional[str] = None) -> List[str]:
        """
        Display edges in a human-readable format.
        
        Args:
            limit: Maximum number of edges to display
            edge_type: Filter by edge type (optional)
            
        Returns:
            List of edge display strings
        """
        displays = []
        count = 0
        
        for edge in self.edges:
            if edge_type and not isinstance(edge, eval(edge_type)):
                continue
                
            if count >= limit:
                break
                
            # 使用边对象的友好显示方法
            displays.append(str(edge))  # 这会调用 __repr__ 方法
            count += 1
        
        return displays
    
    def get_author_collaborators(self, author_name: str) -> List[Dict]:
        """
        Get all collaborators of a specific author.
        
        Args:
            author_name: Name of the author
            
        Returns:
            List of collaborator information
        """
        if author_name not in self.authors:
            return []
        
        author = self.authors[author_name]
        collaborators = []
        
        # 查找所有合作者边
        for edge in self.edges:
            if isinstance(edge, AuthorCoauthorEdge):
                if edge.source == author or edge.target == author:
                    coauthor = edge.target if edge.source == author else edge.source
                    collaborators.append({
                        'name': coauthor.Name,
                        'papers_count': len(edge.attributes.get('coauthored_paper_list', [])),
                        'display': edge.get_simple_display()
                    })
        
        return collaborators
    
    def get_paper_network(self, paper_title: str) -> Dict:
        """
        Get the network information for a specific paper.
        
        Args:
            paper_title: Title of the paper
            
        Returns:
            Dictionary containing paper network information
        """
        if paper_title not in self.papers:
            return {}
        
        paper = self.papers[paper_title]
        network = {
            'title': paper_title,
            'authors': [],
            'affiliations': [],
            'entities': [],
            'citations': {'citing': [], 'cited_by': []}
        }
        
        # 遍历所有边来收集信息
        for edge in self.edges:
            if isinstance(edge, AuthorPaperEdge) and edge.target == paper:
                network['authors'].append({
                    'name': edge.source.Name,
                    'order': edge.attributes.get('author_order', 0),
                    'display': edge.get_simple_display()
                })
            elif isinstance(edge, PaperAffiliationEdge) and edge.source == paper:
                network['affiliations'].append({
                    'name': edge.target.Name,
                    'display': edge.get_simple_display()
                })
            elif isinstance(edge, PaperEntityEdge) and edge.source == paper:
                network['entities'].append({
                    'name': edge.target.name,
                    'weight': edge.attributes.get('weight', 0),
                    'display': edge.get_simple_display()
                })
            elif isinstance(edge, PaperCitationEdge):
                if edge.source == paper:
                    network['citations']['citing'].append({
                        'title': edge.target.Title,
                        'display': edge.get_simple_display()
                    })
                elif edge.target == paper:
                    network['citations']['cited_by'].append({
                        'title': edge.source.Title,
                        'display': edge.get_simple_display()
                    })
        
        # 按作者顺序排序
        network['authors'].sort(key=lambda x: x['order'])
        
        return network
    
    def export_edges_to_json(self, filename: str):
        """
        Export all edges to a JSON file with human-readable format.
        
        Args:
            filename: Output filename
        """
        edges_data = []
        
        for edge in self.edges:
            edge_dict = edge.to_dict()
            edge_dict['display'] = edge.get_simple_display()
            edge_dict['type'] = edge.__class__.__name__
            edges_data.append(edge_dict)
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)
        
        print(f"Edges exported to {filename}")
    
    def print_graph_summary(self):
        """
        Print a human-readable summary of the graph.
        """
        stats = self.get_graph_statistics()
        
        print("\n" + "="*50)
        print("Knowledge Graph Summary")
        print("="*50)
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Edges: {stats['total_edges']}")
        print(f"Connected: {stats['is_connected']}")
        print(f"Components: {stats['number_of_components']}")
        
        print("\nNode Distribution:")
        for node_type, count in stats['node_types'].items():
            print(f"  {node_type}: {count}")
        
        print("\nEdge Distribution:")
        for edge_type, count in stats['edge_types'].items():
            print(f"  {edge_type}: {count}")
        
        print("\nSample Edges (first 5):")
        for display in self.display_edges(limit=5):
            print(f"  {display}")
        
        print("="*50 + "\n")
    
    def export_to_json(self, filename: str):
        """
        Export the graph to a JSON file in node-link format.
        
        Args:
            filename: Output filename (should end with .json)
        """
        # Create a copy of the graph for export to avoid modifying the original
        export_graph = self.graph.copy()
        
        # Prepare nodes for serialization
        for node, data in export_graph.nodes(data=True):
            if 'node_object' in data:
                del data['node_object']
            
        # Prepare edges for serialization
        for u, v, key, data in export_graph.edges(keys=True, data=True):
            if 'edge_object' in data:
                del data['edge_object']
            
            # Convert Paper objects in lists to string titles
            for paper_list_key in ['coauthored_papers', 'collaboration_papers', 'coauthored_paper_list', 'collaboration_paper_list']:
                if paper_list_key in data and data[paper_list_key]:
                    # Check if the list contains Paper objects
                    if hasattr(data[paper_list_key][0], 'Title'):
                        data[paper_list_key] = [p.Title for p in data[paper_list_key]]

        # Generate data in a format suitable for JSON (node-link format)
        graph_data = nx.node_link_data(export_graph)
        
        # Ensure the filename has a .json extension
        if not filename.endswith('.json'):
            filename += '.json'
            
        # Write the data to a JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=4, cls=UUIDEncoder)
            
        print(f"Graph exported to {filename}")
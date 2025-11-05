import json
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from pathlib import Path
import pandas as pd


class KnowledgeGraphAnalyzer:
    """
    知识图谱分析器类 - 用于处理和分析知识图谱JSON文件
    Knowledge Graph Analyzer - for processing and analyzing KG JSON files
    """
    
    def __init__(self, json_file_path: Optional[str] = None):
        """
        初始化知识图谱分析器
        
        Args:
            json_file_path: JSON文件路径
        """
        self.graph_data = None
        self.nx_graph = nx.MultiDiGraph()  # NetworkX多重有向图
        
        # 节点缓存 (Node caches)
        self.papers: Dict[str, Dict] = {}
        self.authors: Dict[str, Dict] = {}
        self.affiliations: Dict[str, Dict] = {}
        self.entities: Dict[str, Dict] = {}
        
        # 边缓存 (Edge caches)
        self.edges_by_type: Dict[str, List[Dict]] = defaultdict(list)
        
        if json_file_path:
            self.load_from_json(json_file_path)
    
    def load_from_json(self, json_file_path: str):
        """
        从JSON文件加载知识图谱数据
        Load knowledge graph data from JSON file
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.graph_data = json.load(f)
        
        self._parse_nodes()
        self._parse_edges()
        self._build_networkx_graph()
        
    def _parse_nodes(self):
        """解析所有节点并按类型分类"""
        if not self.graph_data:
            return
        
        for node in self.graph_data.get('nodes', []):
            node_type = node.get('node_type')
            node_id = node.get('id')
            
            if node_type == 'Paper':
                self.papers[node_id] = node
            elif node_type == 'Author':
                self.authors[node_id] = node
            elif node_type == 'Affiliation':
                self.affiliations[node_id] = node
            elif node_type == 'Entity':
                self.entities[node_id] = node
    
    def _parse_edges(self):
        """解析所有边并按类型分类"""
        if not self.graph_data:
            return
        
        for edge in self.graph_data.get('links', []):
            edge_type = edge.get('edge_type')
            self.edges_by_type[edge_type].append(edge)
    
    def _build_networkx_graph(self):
        """构建NetworkX图结构"""
        if not self.graph_data:
            return
        
        # 添加节点
        for node in self.graph_data.get('nodes', []):
            self.nx_graph.add_node(node['id'], **node)
        
        # 添加边
        for edge in self.graph_data.get('links', []):
            edge_attributes = edge.copy()
            
            source = edge_attributes.pop('source')
            target = edge_attributes.pop('target')
            key = edge_attributes.pop('key', 0) 
            
            self.nx_graph.add_edge(
                source,
                target,
                key=key,
                **edge_attributes  # 将剩余的属性解包传入
            )

    # ==================== 查询方法 (Query Methods) ====================
    
    def get_all_papers(self) -> List[Dict]:
        """获取所有论文"""
        return list(self.papers.values())
    
    def get_all_authors(self) -> List[Dict]:
        """获取所有作者"""
        return list(self.authors.values())
    
    def get_all_affiliations(self) -> List[Dict]:
        """获取所有机构"""
        return list(self.affiliations.values())
    
    def get_all_entities(self) -> List[Dict]:
        """获取所有实体"""
        return list(self.entities.values())
    
    def get_paper_by_title(self, title: str) -> Optional[Dict]:
        """通过标题获取论文"""
        return self.papers.get(title)
    
    def get_author_by_name(self, name: str) -> Optional[Dict]:
        """通过姓名获取作者"""
        for author_id, author in self.authors.items():
            if author.get('name') == name:
                return author
        return None
    
    def get_paper_authors(self, paper_title: str) -> List[Dict]:
        """
        获取论文的所有作者
        Get all authors of a paper
        """
        authors = []
        for edge in self.edges_by_type['AuthorPaper']:
            if edge['target'] == paper_title:
                author_id = edge['source']
                author = self.authors.get(author_id)
                if author:
                    author_info = author.copy()
                    author_info['order'] = edge.get('author_order', 0)
                    authors.append(author_info)
        
        # 按作者顺序排序
        authors.sort(key=lambda x: x.get('order', 999))
        return authors
    
    def get_paper_affiliations(self, paper_title: str) -> List[Dict]:
        """获取论文相关的所有机构"""
        affiliations = []
        for edge in self.edges_by_type['PaperAffiliation']:
            if edge['source'] == paper_title:
                affiliation_id = edge['target']
                affiliation = self.affiliations.get(affiliation_id)
                if affiliation:
                    affiliations.append(affiliation)
        return affiliations
    
    def get_paper_entities(self, paper_title: str) -> List[Dict]:
        """获取论文研究的所有实体"""
        entities = []
        for edge in self.edges_by_type['PaperEntity']:
            if edge['source'] == paper_title:
                entity_id = edge['target']
                entity = self.entities.get(entity_id)
                if entity:
                    entity_info = entity.copy()
                    entity_info['weight'] = edge.get('weight', 0)
                    entities.append(entity_info)
        return entities
    
    def get_author_papers(self, author_name: str) -> List[Dict]:
        """获取作者的所有论文"""
        author = self.get_author_by_name(author_name)
        if not author:
            return []
        
        papers = []
        author_id = author['id']
        for edge in self.edges_by_type['AuthorPaper']:
            if edge['source'] == author_id:
                paper_title = edge['target']
                paper = self.papers.get(paper_title)
                if paper:
                    paper_info = paper.copy()
                    paper_info['author_order'] = edge.get('author_order', 0)
                    papers.append(paper_info)
        return papers
    
    def get_author_coauthors(self, author_name: str) -> List[Dict]:
        """
        获取作者的所有合作者
        Get all coauthors of an author
        """
        author = self.get_author_by_name(author_name)
        if not author:
            return []
        
        coauthors = []
        author_id = author['id']
        
        for edge in self.edges_by_type['Coauthor']:
            coauthor_id = None
            if edge['source'] == author_id:
                coauthor_id = edge['target']
            elif edge['target'] == author_id:
                coauthor_id = edge['source']
            
            if coauthor_id:
                coauthor = self.authors.get(coauthor_id)
                if coauthor:
                    coauthor_info = coauthor.copy()
                    coauthor_info['coauthored_papers'] = edge.get('coauthored_papers', [])
                    coauthors.append(coauthor_info)
        
        return coauthors
    
    def get_affiliation_collaborations(self, affiliation_name: str) -> List[Dict]:
        """获取机构的合作关系"""
        collaborations = []
        
        for edge in self.edges_by_type['AffiliationCollaboration']:
            if edge['source'] == affiliation_name or edge['target'] == affiliation_name:
                other_affiliation = edge['target'] if edge['source'] == affiliation_name else edge['source']
                collab_info = {
                    'affiliation': other_affiliation,
                    'collaboration_papers': edge.get('collaboration_papers', [])
                }
                collaborations.append(collab_info)
        
        return collaborations
    
    def edges_of_entity(self, entity_display_name: str) -> List[Dict]:
        """
        根据实体的显示名称获取与之连接的所有边, 包含边的全部信息
        Get all edges connected to an entity by its display name, with full edge information.
        """
        entity_id = None
        if not self.graph_data or 'nodes' not in self.graph_data:
            return []

        # Find the entity's ID from its display name
        for node in self.graph_data['nodes']:
            # Check for 'name' or 'display_name' or even 'title' for papers
            if node.get('name') == entity_display_name or \
               node.get('display_name') == entity_display_name or \
               node.get('title') == entity_display_name:
                entity_id = node.get('id')
                break
        
        if not entity_id:
            # If no node is found, return an empty list
            return []

        connected_edges = []
        if 'links' not in self.graph_data:
            return connected_edges

        for edge in self.graph_data['links']:
            if edge.get('source') == entity_id or edge.get('target') == entity_id:
                connected_edges.append(edge)
        
        return connected_edges
    
    # ==================== 统计方法 (Statistics Methods) ====================
    
    def get_statistics(self) -> Dict:
        """
        获取知识图谱的整体统计信息
        Get overall statistics of the knowledge graph
        """
        stats = {
            'total_nodes': len(self.graph_data.get('nodes', [])), # pyright: ignore[reportOptionalMemberAccess]
            'total_edges': len(self.graph_data.get('links', [])), # pyright: ignore[reportOptionalMemberAccess]
            'node_counts': {
                'papers': len(self.papers),
                'authors': len(self.authors),
                'affiliations': len(self.affiliations),
                'entities': len(self.entities)
            },
            'edge_counts': {edge_type: len(edges) 
                          for edge_type, edges in self.edges_by_type.items()},
            'is_directed': self.graph_data.get('directed', False), # pyright: ignore[reportOptionalMemberAccess]
            'is_multigraph': self.graph_data.get('multigraph', False) # pyright: ignore[reportOptionalMemberAccess]
        }
        return stats
    
    def get_author_statistics(self, author_name: str) -> Dict:
        """获取作者的统计信息"""
        papers = self.get_author_papers(author_name)
        coauthors = self.get_author_coauthors(author_name)
        author = self.get_author_by_name(author_name)
        
        if not author:
            return {}
        
        # 获取作者的所属机构
        affiliations = []
        author_id = author['id']
        for edge in self.edges_by_type['AuthorAffiliation']:
            if edge['source'] == author_id:
                affiliations.append(edge['target'])
        
        return {
            'name': author_name,
            'email': author.get('email'),
            'paper_count': len(papers),
            'coauthor_count': len(coauthors),
            'affiliations': affiliations,
            'papers': [p['title'] for p in papers]
        }
    
    def get_paper_statistics(self, paper_title: str) -> Dict:
        """获取论文的统计信息"""
        paper = self.get_paper_by_title(paper_title)
        if not paper:
            return {}
        
        authors = self.get_paper_authors(paper_title)
        affiliations = self.get_paper_affiliations(paper_title)
        entities = self.get_paper_entities(paper_title)
        
        return {
            'title': paper_title,
            'field': paper.get('field'),
            'author_count': len(authors),
            'affiliation_count': len(affiliations),
            'entity_count': len(entities),
            'authors': [a['name'] for a in authors],
            'affiliations': [a['name'] for a in affiliations],
            'entities': [e['name'] for e in entities]
        }
    

    # ==================== 导出方法 (Export Methods) ====================
    
    def export_to_csv(self, output_dir: str = "."):
        """
        导出数据到CSV文件
        Export data to CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 导出论文
        papers_df = pd.DataFrame(self.get_all_papers())
        papers_df.to_csv(output_path / "papers.csv", index=False, encoding='utf-8-sig')
        
        # 导出作者
        authors_df = pd.DataFrame(self.get_all_authors())
        authors_df.to_csv(output_path / "authors.csv", index=False, encoding='utf-8-sig')
        
        # 导出机构
        affiliations_df = pd.DataFrame(self.get_all_affiliations())
        affiliations_df.to_csv(output_path / "affiliations.csv", index=False, encoding='utf-8-sig')
        
        # 导出实体
        entities_df = pd.DataFrame(self.get_all_entities())
        entities_df.to_csv(output_path / "entities.csv", index=False, encoding='utf-8-sig')
        
        print(f"数据已导出到 {output_path}")
    



# 使用示例 (Usage Example)
if __name__ == "__main__":
    # 初始化分析器
    analyzer = KnowledgeGraphAnalyzer("knowledge_graph.json")
    
    # 获取统计信息
    stats = analyzer.get_statistics()
    print("知识图谱统计:", stats)
    
    # 查询某篇论文的作者
    paper_title = "GraphRAG: A Framework for Automatic Agent Generation"
    authors = analyzer.get_paper_authors(paper_title)
    print(f"\n论文 '{paper_title}' 的作者:", [a['name'] for a in authors])
    
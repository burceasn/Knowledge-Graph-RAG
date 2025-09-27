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
        
        # 添加边 (已修复)
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
    
    # ==================== 网络分析方法 (Network Analysis Methods) ====================
    
    def get_most_productive_authors(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        获取最高产的作者 (按论文数量)
        Get most productive authors by paper count
        """
        author_paper_counts = defaultdict(int)
        
        for edge in self.edges_by_type['AuthorPaper']:
            author_id = edge['source']
            author = self.authors.get(author_id)
            if author:
                author_paper_counts[author['name']] += 1
        
        sorted_authors = sorted(author_paper_counts.items(), 
                               key=lambda x: x[1], reverse=True)
        return sorted_authors[:top_n]
    
    def get_most_collaborative_affiliations(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """获取合作最多的机构"""
        affiliation_collab_counts = defaultdict(int)
        
        for edge in self.edges_by_type['AffiliationCollaboration']:
            affiliation_collab_counts[edge['source']] += 1
            affiliation_collab_counts[edge['target']] += 1
        
        sorted_affiliations = sorted(affiliation_collab_counts.items(),
                                    key=lambda x: x[1], reverse=True)
        return sorted_affiliations[:top_n]
    
    def get_research_hotspots(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        获取研究热点 (基于实体出现频率)
        Get research hotspots based on entity frequency
        """
        entity_counts = defaultdict(int)
        
        for edge in self.edges_by_type['PaperEntity']:
            entity_id = edge['target']
            entity = self.entities.get(entity_id)
            if entity:
                entity_counts[entity['name']] += 1
        
        sorted_entities = sorted(entity_counts.items(),
                                key=lambda x: x[1], reverse=True)
        return sorted_entities[:top_n]
    
    def find_shortest_path_between_authors(self, author1_name: str, author2_name: str) -> List:
        """
        寻找两个作者之间的最短路径
        Find shortest path between two authors
        """
        author1 = self.get_author_by_name(author1_name)
        author2 = self.get_author_by_name(author2_name)
        
        if not author1 or not author2:
            return []
        
        try:
            path = nx.shortest_path(self.nx_graph.to_undirected(), 
                                   author1['id'], author2['id'])
            return path
        except nx.NetworkXNoPath:
            return []
    
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
    
    def generate_report(self) -> str:
        """
        生成知识图谱分析报告
        Generate knowledge graph analysis report
        """
        stats = self.get_statistics()
        top_authors = self.get_most_productive_authors(5)
        top_affiliations = self.get_most_collaborative_affiliations(5)
        hotspots = self.get_research_hotspots(5)
        
        report = []
        report.append("="*50)
        report.append("知识图谱分析报告 (Knowledge Graph Analysis Report)")
        report.append("="*50)
        
        report.append("\n## 基本统计 (Basic Statistics)")
        report.append(f"- 总节点数 (Total Nodes): {stats['total_nodes']}")
        report.append(f"- 总边数 (Total Edges): {stats['total_edges']}")
        report.append(f"- 论文数 (Papers): {stats['node_counts']['papers']}")
        report.append(f"- 作者数 (Authors): {stats['node_counts']['authors']}")
        report.append(f"- 机构数 (Affiliations): {stats['node_counts']['affiliations']}")
        report.append(f"- 实体数 (Entities): {stats['node_counts']['entities']}")
        
        report.append("\n## 边类型分布 (Edge Type Distribution)")
        for edge_type, count in stats['edge_counts'].items():
            report.append(f"- {edge_type}: {count}")
        
        report.append("\n## Top 5 高产作者 (Most Productive Authors)")
        for i, (author, count) in enumerate(top_authors, 1):
            report.append(f"{i}. {author}: {count} 篇论文")
        
        report.append("\n## Top 5 合作机构 (Most Collaborative Affiliations)")
        for i, (affiliation, count) in enumerate(top_affiliations, 1):
            report.append(f"{i}. {affiliation}: {count} 个合作关系")
        
        report.append("\n## Top 5 研究热点 (Research Hotspots)")
        for i, (entity, count) in enumerate(hotspots, 1):
            report.append(f"{i}. {entity}: 出现 {count} 次")
        
        return "\n".join(report)


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
    
    # 生成报告
    report = analyzer.generate_report()
    print("\n", report)
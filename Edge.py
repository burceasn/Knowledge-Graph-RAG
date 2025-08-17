import uuid
from typing import Dict, Any, Type, Optional
from Node import Author, Paper, Affiliation, Entity

class BaseEdge:
    """
    知识图谱中所有边的基类。

    Attributes:
        _id (uuid.UUID): 边的唯一标识符。
        source (Any): 边的源节点。
        target (Any): 边的目标节点。
        relation (str): 描述关系类型的字符串。
        attributes (Dict[str, Any]): 存储边特定属性的字典。
    """
    def __init__(self, source_node: Any, target_node: Any, relation: str, **kwargs):
        """
        初始化一个边的实例。

        Args:
            source_node: 源节点对象。
            target_node: 目标节点对象。
            relation: 关系的名称。
            **kwargs: 边的其他属性。
        """
        self._id = uuid.uuid4()
        self.source = source_node
        self.target = target_node
        self.relation = relation
        self.attributes = kwargs

    def __repr__(self) -> str:
        """返回一个可读的边表示。"""
        source_type = self.source.__class__.__name__
        target_type = self.target.__class__.__name__
        return f"Edge(id={self._id}, {source_type} -[{self.relation}]-> {target_type})"

    def to_dict(self) -> Dict[str, Any]:
        """将边对象序列化为字典。"""
        return {
            'id': str(self._id),
            'source_type': self.source.__class__.__name__,
            'target_type': self.target.__class__.__name__,
            'relation': self.relation,
            'attributes': self.attributes
        }

#? 怎么解决不同的author和同一篇paper不同的weight --> 通过记录author的order来区分
#! weight的算法放在之后的文件里面
class AuthorPaperEdge(BaseEdge):
    """定义“作者-撰写->论文”的边。"""
    def __init__(self, author: Author, paper: Paper, author_order: int = 0):
        super().__init__(
            source_node=author,
            target_node=paper,
            relation="writes",
            author_order=author_order,
            weight = 0.0 # 初始权重为0.0
        )
    
    def update_weight(self, new_weight: int):
        self.attributes['weight'] = new_weight

#! 定义了一个Rank属性, 用来规范author再affiliation中的顺序, 具体算法之后规定
class AuthorAffiliationEdge(BaseEdge):
    """定义“作者-从属于->机构”的边。"""
    def __init__(self, author: Author, affiliation: Affiliation):
        super().__init__(
            source_node=author,
            target_node=affiliation,
            relation="is_affiliated_with",
            rank = 0 # 初始化排名为0
        )

    def update_rank(self, new_rank: int):
        """更新作者在机构中的排名。"""
        self.attributes['rank'] = new_rank  


class AuthorCoauthor(BaseEdge):
    """定义“作者-合作者->作者”的边。"""
    def __init__(self, author1: Author, author2: Author, coauthored_papers: int = 1):
        # 为避免重复，可以约定一个顺序，例如基于哈希值
        if hash(author1) > hash(author2):
            author1, author2 = author2, author1
            
        super().__init__(
            source_node=author1,
            target_node=author2,
            relation="coauthor_of",
            weight = 0.0,  # 初始权重为0.0
            coauthored_papers=coauthored_papers
        )

    def update_weight(self, new_weight: int):
        """更新合作者之间的权重。"""
        self.attributes['weight'] = new_weight
  

import uuid
import json
from typing import Dict, Any, List, Type, Optional
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
            'relation': f"**{self.relation}**",  # 加粗处理, 发现LLM真的理解加粗的内容
            'attributes': self.attributes
        }

    def to_json(self, **kwargs) -> str:
        """将边对象序列化为JSON字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

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
    
    def update_weight(self, new_weight: float):
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


class AuthorCoauthorEdge(BaseEdge):
    """定义“作者-合作者->作者”的边。"""
    def __init__(self, author1: Author, author2: Author, coauthored_paper: Paper):      # 建立合作者关系时，必须指定一篇共同撰写的论文
        # 为避免重复，可以约定一个顺序，例如基于哈希值
        if hash(author1) > hash(author2):
            author1, author2 = author2, author1
            
        super().__init__(
            source_node=author1,
            target_node=author2,
            relation="coauthor_of",
            weight = 0.0,  # 初始权重为0.0
            coauthored_paper_list=[coauthored_paper]  # 用list表示
        )

    def update_weight(self, new_weight: float):
        """更新合作者之间的权重。"""
        self.attributes['weight'] = new_weight

    def add_coauthored_paper(self, new_paper: Paper):
        """添加一篇新的共同撰写的论文。"""
        if 'coauthored_paper_list' not in self.attributes:
            self.attributes['coauthored_paper_list'] = []
        if new_paper not in self.attributes['coauthored_paper_list']:
            self.attributes['coauthored_paper_list'].append(new_paper)


# 这个edge感觉关心的人比较少, 简单处理了
class PaperAffiliationEdge(BaseEdge):
    """定义“论文-关联->机构”的边。"""
    def __init__(self, paper: Paper, affiliation: Affiliation):
        super().__init__(
            source_node=paper,
            target_node=affiliation,
            relation="is_associated_with"
        )


# paper citation已经有很多现成的工具了, 这个就不细讲了
class PaperCitationEdge(BaseEdge):
    """定义“论文-引用->论文”的边。"""
    def __init__(self, citing_paper: Paper, cited_paper: Paper):
        super().__init__(
            source_node=citing_paper,
            target_node=cited_paper,
            relation="cites"
        )


class PaperEntityEdge(BaseEdge):
    """定义“论文-提及->实体”的边。"""
    def __init__(self, paper: Paper, entity: Entity):
        super().__init__(
            source_node=paper,
            target_node=entity,
            relation="researchs",
            weight = 0.0 # 相当于importance, 初始化为0.0
        )
    
    # 这个nano_graphrag里面有weight, 可以直接用maybe
    def update_weight(self, new_weight: float):
        """更新论文与实体之间的权重。"""
        self.attributes['weight'] = new_weight


class AffiliationCollaborationEdge(BaseEdge):
    """定义“机构-合作->机构”的边。"""
    def __init__(self, affiliation1: Affiliation, affiliation2: Affiliation, collaboration_paper: Paper):
        # 为避免重复，可以约定一个顺序，例如基于哈希值
        if hash(affiliation1) > hash(affiliation2):
            affiliation1, affiliation2 = affiliation2, affiliation1
            
        super().__init__(
            source_node=affiliation1,
            target_node=affiliation2,
            relation="collaborates_with",
            weight = 0.0,  # 初始权重为0.0
            collaboration_paper_list=[collaboration_paper]  # 用list表示
        )

    def update_weight(self, new_weight: float):
        """更新机构之间的权重。"""
        self.attributes['weight'] = new_weight

    def add_collaboration_paper(self, new_paper: Paper):
        """添加一篇新的合作论文。"""
        if 'collaboration_paper_list' not in self.attributes:
            self.attributes['collaboration_paper_list'] = []
        if new_paper not in self.attributes['collaboration_paper_list']:
            self.attributes['collaboration_paper_list'].append(new_paper)
            
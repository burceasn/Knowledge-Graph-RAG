"""
Paper Entity Extractor — 同步版本 (synchronous)
从论文摘要中提取实体和关系，构建知识图谱
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import networkx as nx
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# 导入您的 Node 和 Edge 类
from Node import Paper, Entity
from Edge import PaperEntityEdge, EntityToEntityEdge

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("./prompts/entity_and_relationship_extraction_prompt.txt", "r", encoding="utf-8") as f:
    ENTITY_EXTRACTION_PROMPT = f.read()

DEFAULT_ENTITY_TYPES = ["CONCEPT", "METHOD", "DATASET", "METRIC", "APPLICATION", "TOOL", "PROBLEM", "RESULT", "FRAMEWORK"]

@dataclass
class ExtractedEntity:
    name: str
    type: str
    description: str

@dataclass
class ExtractedRelation:
    source: str
    target: str
    relationship: str
    description: str

class PaperEntityExtractor:
    """
    实体提取器（synchronous）
    """
    def __init__(
        self,
        llm_provider: str = "ollama",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model: str = "llama3",
        entity_types: List[str] = [],
        # tuple_delimiter: str = "<|>",
        record_delimiter: str = "##",
        max_tokens: int = 4000,
        temperature: float = 0.1
    ):
        self.llm_provider = llm_provider
        self.model = model
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        # self.tuple_delimiter = tuple_delimiter
        self.record_delimiter = record_delimiter
        self.max_tokens = max_tokens
        self.temperature = temperature

        # 初始化同步 LLM 客户端（OpenAI）
        if llm_provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o-mini"
        elif llm_provider == "deepseek":
            # deepseek 通过 base_url 指定
            self.client = OpenAI(api_key=api_key, base_url=api_base_url or "https://api.deepseek.com/beta")
            self.model = "deepseek-chat"
        elif llm_provider == "ollama":
            self.client = OpenAI(api_key="ollama", base_url=api_base_url or "http://localhost:11434/v1")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # NetworkX 图
        self.graph = nx.Graph()

    def _get_response_content(self, response) -> str:
        """
        从 OpenAI 响应中稳健地抽取文本内容（兼容不同客户端返回结构）
        """
        if self.llm_provider == "deepseek":
            try:
                # For DeepSeek with function calling, handle the response as a dict or object
                if hasattr(response, 'model_dump'):
                    response_data = response.model_dump()
                else:
                    response_data = dict(response)
                
                # Check if this is a function call response
                choices = response_data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    tool_calls = message.get("tool_calls", [])
                    
                    if tool_calls:
                        # This is a function call response
                        return json.dumps(response_data)
                    else:
                        # Regular text response (fallback)
                        content = message.get("content", "")
                        if content:
                            return content
                
                # If no valid response found, return empty string
                logger.warning("No valid content found in DeepSeek response")
                return json.dumps(response_data)  # Return full response for debugging
                
            except Exception as e:
                logger.error(f"Error processing DeepSeek response: {e}")
                # Return the response as string for debugging
                return str(response)
        
        else:
            # Original logic for other providers
            try:
                # 可能的属性链：response.choices[0].message.content
                choices = getattr(response, "choices", None)
                if choices:
                    first = choices[0]
                    msg = getattr(first, "message", None)
                    if msg:
                        content = getattr(msg, "content", None)
                        if content:
                            return content
                    # 或者 first["message"]["content"]
                    try:
                        return first["message"]["content"]
                    except Exception:
                        pass
                # 也尝试 dict 风格
                if isinstance(response, dict):
                    return response["choices"][0]["message"]["content"]
            except Exception:
                logger.exception("Failed to extract content from response object.")
            # 最后兜底
            return str(response)

    def extract_entities_from_abstract(self, paper: Paper) -> tuple[List[Entity], List[EntityToEntityEdge]]:
        """
        同步地从单篇论文摘要提取实体和关系
        """
        abstract = paper.Abstract
        if not abstract:
            logger.warning(f"Paper '{paper.Title}' has no abstract.")
            return [], []

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            input_text=abstract
        )

        if self.llm_provider == "deepseek":
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "extract_entities_and_relations",
                        "description": "extract all the entities and the relationships between the entities from the given source. The entities and the relationships should match the given schema. All of the entities and the relationships should be based on the given source, and you should not miss any important entities or relationships. You should not make up any entities or relationships that are not in the source.",
                        "strict": True, 
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "entities": {
                                    "type": "array",
                                    "description": "A list of entities extracted from the source.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "entity_name": {
                                                "type": "string",
                                                "description": "The name of the entity, capitalized."
                                            },
                                            "entity_type": {
                                                "type": "string",
                                                "description": f"The type of the entity. Must be one of the following types: {self.entity_types}.",
                                                "enum": self.entity_types 
                                            },
                                            "entity_description": {
                                                "type": "string",
                                                "description": "A comprehensive description of the entity's attributes and activities."
                                            }
                                        },
                                        "required": ["entity_name", "entity_type", "entity_description"]
                                    }
                                },
                                "relationships": {
                                    "type": "array",
                                    "description": "Identify the explicit relationships between the extracted entities. Only include relationships between the entities in the entities list.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "source": {
                                                "type": "string",
                                                "description": "the name of the source entity, must match one of the entity_name in the entities list."
                                            },
                                            "target": {
                                                "type": "string",
                                                "description": "the name of the target entity, must match one of the entity_name in the entities list, and cannot be the same as the source."
                                            },
                                            "relationship_description": {
                                                "type": "string",
                                                "description": "a brief and comprehensive description of the relationship between the source and target entities."
                                            },
                                            "relationship_strength": {
                                                "type": "integer",
                                                "description": "based on the given source, how strong is the relationship between the source and target entities, on a scale of 1 to 10, where 1 is very weak and 10 is very strong.",
                                                "minimum": 1,
                                                "maximum": 10
                                            }
                                        },
                                        "required": ["source", "target", "relationship_description", "relationship_strength"]
                                    }
                                }
                            },
                            "required": ["entities", "relationships"]
                        }
                    }
                }
            ]
            
            try:
                response = self.client.chat.completions.create( # pyright: ignore[reportAttributeAccessIssue]
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting entities and relationships from academic text. You don't make any mistakes, and you don't miss anything important."},
                        {"role": "user", "content": prompt}
                    ],
                    tools=tools, # pyright: ignore[reportArgumentType]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Use _get_response_content to handle the response
                content = self._get_response_content(response)
                
                # Parse the response using the updated parsing function
                entities, relations = self._parse_extraction_response(content)
                
            except Exception as e:
                logger.error(f"Error calling DeepSeek API: {e}")
                return [], []
                
        else:
            # Handle other LLM providers
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting entities and relationships from academic text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = self._get_response_content(response)
                entities, relations = self._parse_extraction_response(content)
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
                return [], []
        
        return entities, relations

    def _parse_extraction_response(self, response: Union[str, dict]) -> tuple[List[Entity], List[EntityToEntityEdge]]:
        entities: List[Entity] = []
        relations: List[EntityToEntityEdge] = []
        entity_dict: Dict[str, Entity] = {}  # 用于查找实体对象

        if self.llm_provider == "deepseek":
            try:
                if isinstance(response, str):
                    response_data = json.loads(response)
                else:
                    response_data = response

                tool_calls = response_data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                
                if tool_calls:
                    # Get the function arguments (which is a JSON string)
                    arguments_str = tool_calls[0].get("function", {}).get("arguments", "{}")
                    
                    # Parse the arguments JSON string
                    extracted_data = json.loads(arguments_str)
                    
                    # Process entities
                    for entity_data in extracted_data.get("entities", []):
                        entity = Entity(
                            name=entity_data["entity_name"],
                            description=entity_data["entity_description"],
                            entity_type=entity_data["entity_type"]
                        )
                        entities.append(entity)
                        entity_dict[entity.name] = entity  # Add entity to lookup dictionary
                        logger.debug(f"Extracted entity: {entity.name} ({entity.entity_type})")
                    
                    # Process relationships
                    for rel_data in extracted_data.get("relationships", []):
                        source_name = rel_data["source"]
                        target_name = rel_data["target"]
                        if source_name in entity_dict and target_name in entity_dict:
                            relation = EntityToEntityEdge(
                                source_entity=entity_dict[source_name],
                                target_entity=entity_dict[target_name],
                                relationship_description=rel_data["relationship_description"],
                                strength=rel_data["relationship_strength"] / 10.0  # 转换为 0-1 的范围
                            )
                            relations.append(relation)
                        else:
                            logger.warning(f"Relationship references unknown entity: {source_name} -> {target_name}")
                    
                    logger.info(f"Successfully extracted {len(entities)} entities and {len(relations)} relationships")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
            except Exception as e:
                logger.error(f"Error parsing DeepSeek response: {e}")
                logger.debug(f"Response structure: {response}")
        
        else:
                """
                我发现其实拿deepseek的模型试一试很快, 暂时不弄其他模型的事情
                """
                pass
        
        return entities, relations


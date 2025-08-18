'''
生产集成模块
'''
import logging
import os 
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    '''
    集成LLM与回答生成
    '''
    def __init__(self,model_name:str,temperature:float = 0,max_tokens:int = 2048):
        '''
        初始化
        '''
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self._setup_llm()

    def _setup_llm(self):
        '''
        加载llm
        '''
        logger.info(f"正在加载LLM : {self.model_name}")
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("请先设置API_KEY")
        self.llm = ChatOpenAI(
            model = self.model_name,
            temperature = self.temperature,
            api_key = api_key,
            streaming = True,   #允许流式输出
            base_url = os.getenv("BASE_URL"),
            max_tokens = self.max_tokens
        )
        logger.info(f"LLM加载完成")
    
    def generate_basic_answer(self,query:str,context:List[Document])->str:
        '''
        根据检索到的上下文,让LLM生成基础回答
        Args:
            query:用户查询
            context:检索到的上下文
        Returns:
            LLM回答
        '''
        context = self._build_context(context)
        prompt = ChatPromptTemplate.from_template(
'''
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:         
'''
        )
        #构建链
        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        response = chain.invoke({'question':query,'context':context})
        return response
    
    def _build_context(self,context:List[Document],max_length = 2000)->str:
        '''
        构建上下文，负责把文档列表拼成一个整体字符串
        Args:
            context: 上下文文档集合
            max_length: 最大长度
        Returns:
            格式化后的上下文字符串
        '''
        if not context:
            return "暂无上下文信息"
        #初始化一个容器，存放处理后的上下文，同时记录长度
        context_parts = []
        current_length = 0
        for i,doc in enumerate(context,1):  #从1开始计数

            #首先取出元数据
            metadata_info = f"食谱{i}"
            if 'dish_name' in doc.metadata:
                metadata_info += f"{doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f"| 分类：{doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f"| 难易程度：{doc.metadata['difficulty']}"

            #构建文本，拼接起来
            doc_context = f"{metadata_info} \n {doc.page_content}"

            #检查长度限制
            if current_length + len(doc_context) > max_length:
                break

            context_parts.append(doc_context)
            current_length += len(doc_context)
        return "\n" + '='*50 + '\n'+ '\n'.join(context_parts)  #换行后，每一行放一条文档数据

    def query_rewrite(self,query:str)->str:
        '''
        由大模型来判断是否需要重写质量不高的query
        '''
        prompt = PromptTemplate.from_template(
'''
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:
'''
        )
        #构建链
        chain = (prompt | self.llm | StrOutputParser())

        response = chain.invoke({'query':query}).strip()  #得到输出并移除首尾的空格

        if response != query:
            logger.info(f"查询已改写：'{query}' -> '{response}'")
        else:
            logger.info(f"查询未改写: '{query}")

        return response

    def query_router(self,query:str)->str:
        '''
        实现多查询功能-查询路由
        根据query选择不同的处理方式
        Args:
            query:查询
        Returns:
            路由类型：('list', 'detail', 'general')
        '''
        prompt = ChatPromptTemplate.from_template(
            '''
            根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:'''
        )

        chain = (prompt | self.llm | StrOutputParser())

        response = chain.invoke({'query':query}).strip().lower()

        #检查分类有效性
        if response in ['list', 'detail', 'general']:
            return response
        else:
            return 'general'
        
    def generate_list_answer(self,query:str,context:List[Document])->str:
        '''
        list 类型的问题回答器-适合推荐类查询，推荐一系列菜品
        Args:
            query:提问
            context:上下文
        Returns:
            菜品回答
        '''
        if not context:
            return "很抱歉，没有找到相关的菜谱"

        dishes = []
        for doc in context:
            dish_name = doc.metadata.get("dish_name","未知菜品")
            #去重
            if dish_name not in dishes:
                dishes.append(dish_name)
        
        #构造回答
        if len(dishes) == 1:
            return f"为你推荐此菜品：{dishes[0]} \n"
        elif len(dishes) <= 3:
            return f"为你推荐以下菜品 : \n" + "\n".join( [f"{i},{name}" for i,name in enumerate(dishes)] )
        else:
            return f"为你推荐以下菜品 : \n" + "\n".join( [f"{i},{name}" for i,name in enumerate(dishes[:3])] ) + f"\n\n 还有其他{len(dishes) - 3} 道菜品可供选择"
                
    def generate_detail_answer(self,query:str,context:List[Document]):
        '''
        detail类型的回答器
        Args:
            query:查询
            context:上下文
        Returns:
            LLM回答
        '''
        context = self._build_context(context)
        
        prompt = ChatPromptTemplate.from_template(
        '''你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性

回答:'''
        )
        chain = (prompt | self.llm | StrOutputParser())
        input = {'question':query,'context':context}

        response = chain.invoke(input)
        return response

    def generate_general_answer(self,query:str,context:List[Document]):
        '''
        general类型的生成器
        Args:
            query:查询
            context:上下文
        '''
        context = self._build_context(context)
        prompt = ChatPromptTemplate.from_template(
            '''
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:'''
        )
        chain = (prompt | self.llm | StrOutputParser())

        input = {'question':query,'context':context}

        response = chain.invoke(input)
        return response

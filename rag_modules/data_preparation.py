from langchain_core.documents import Document
from typing import List, Dict, Any #类型注解系统
from pathlib import Path
import uuid
from langchain_text_splitters import MarkdownHeaderTextSplitter
import logging
import hashlib

#获取一个以当前模块名为标识的日志记录器,用于打印信息
logger = logging.getLogger(__name__)

class DataPreparationModule:
    '''
    数据准备模块，负责数据的加载，清洗，预处理
    '''
    def __init__(self,data_path:str) -> None:
        '''
        初始化，参数为文件的路径
        我们使用父子文档的方式进行处理
        检索时使用子文档，保持精确性与效率
        生成时使用对应子文档的父文档，保持内容的完整性
        Args:
            data_path: 数据文件夹路径
        '''
        self.data_path = data_path
        self.documents : List[Document] = [] #父文档
        self.chunks : List[Document] = [] #子文档
        self.parent_child_map : Dict[str,str] = {} #父子文档的hash映射

    def load_documents(self) -> List[Document]:
        '''
        载入文档
        Returns:
            加载的文档列表
        '''
        documents = []
        data_path_obj = Path(self.data_path)

        for md_file in data_path_obj.rglob("*.md"): #递归的读取路径下的所有md文件
            #读取文件内容
            with open(md_file,'r',encoding='utf-8') as f:
                content = f.read()

            #为父文档分配唯一id
            parent_id = hashlib.md5(str(md_file).encode()).hexdigest()

            #创建document对象，document是一个带有“内容”和“元数据”的结构化文档对象
            #设计了三项元数据，分别是文件路径，id，文件类型(父文档/子文档)
            doc = Document(
                page_content = content,
                metadata = {
                    "source":str(md_file),
                    "parent_id" :parent_id,
                    "doc_type":"parent"
                }
            )
            documents.append(doc)
        
        for doc in documents:
            self._enhance_metadata(doc)
        self.documents = documents

        return documents

    def _enhance_metadata(self,doc: Document): #用_开头表示内部使用，不被外部调用
        '''
        增强文档的元数据
        通过文档的路径，补充以下元数据
        1.菜品的分类 'category'
        2.菜品的名称 'dish_name'
        3.菜品难度等级 'difficulty'
        Args:
            需要增强的文档
        '''
        #获取文件的路径
        file_path = Path(doc.metadata.get('source','')) #用get函数获取键值，第二个参数为默认值

        #获得这个路径的每一级目录结构 
        #例如"data/articles/article1.md" ->('data', 'articles', 'article1.md')
        path_parts = file_path.parts

        #提取菜品的分类
        category_mapping = {
            'meat_dish': '荤菜', 'vegetable_dish': '素菜', 'soup': '汤品',
            'dessert': '甜品', 'breakfast': '早餐', 'staple': '主食',
            'aquatic': '水产', 'condiment': '调料', 'drink': '饮品'
        }

        #根据文件的路径，在哪个文件夹里，来获得分类
        #先统一初始化为其他
        doc.metadata['category'] = '其他'
        for k,v in category_mapping.items():
            if k in path_parts:
                doc.metadata['category'] = v
                break
        
        #提取菜品的名称
        doc.metadata['dish_name'] = file_path.stem

        #提取菜品难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等难度'
        elif '★★' in content:
            doc.metadata['difficulty'] = '比较简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '简单'
        else:
            doc.metadata['difficulty'] = '未知难度'

    def chunk_documents(self)->List[Document]:
        '''
        基于Markdown结构感知进行子文档分块
        并且添加对应的元数据
        Returns:
            分块后的子文档
        '''
        #如果父文档未加载，抛出异常
        if not self.documents:
            raise ValueError("未加载文档")
    
        #切割子文档
        chunks = self._markdown_header_split()

        #为子文档添加元数据
        #'chunk_id' 'chunk_size'
        for i,chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        #复制值到self去
        self.chunks = chunks
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        '''
        使用Markdown标题分割器进行结构化分割
        '''
        #定义要分割的标题层级
        headers_to_split_on = [
            ("#","主标题"), #菜品名称
            ("##","二级标题"), #必备原料、计算、操作等
            ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        #创建Markdown分割器
        #这个langchain的Markdown分割器会将每一级标题下的内容作为文本块
        #同时将层级信息作为元数据附加上去
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        all_chunks = []
        for doc in self.documents:
            try:
                #对每个文档进行Markdown分割
                single_chunks = markdown_splitter.split_text(doc.page_content)

                #建立父子文档关系
                #先拿到大的父文档的id
                parent_id = doc.metadata["parent_id"]

                #遍历所有的子块
                for i,chunk in enumerate(single_chunks):
                    #获取一个子文档的标识
                    child_id = str(uuid.uuid4())

                    #先把父文档的元数据拷贝给子文档
                    chunk.metadata.update(doc.metadata)
                    #赋值dish_name
                    chunk.metadata["dish_name"] = doc.metadata.get("dish_name", "未知菜品")
                    #这里我们复制了"source"，"parent_id"，"doc_type"，'category'，'dish_name'，'difficulty'
                    #我们需要更改其中的"doc_type"，并添加一些元数据
                    chunk.metadata.update(
                        {
                            "chunk_id":child_id,
                            "parent_id":parent_id,  
                            "doc_type":"child",#标记为子文档
                            "chunk_index":i #子文档在父文档中的位置
                        
                        }
                    )

                    #再建立父文档子文档的映射关系
                    #因为我们需要根据子文档查找父文档，所以用子文档id作为键,父文档作为值
                    self.parent_child_map[child_id] = parent_id
                #把这一个文件的所有块加入块集合中
                all_chunks.extend(single_chunks)
            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)

        return all_chunks

    def get_parent_document(self,child_chunks:List[Document])->List[Document]:
        '''
        根据子文档查找父文档
        由于找到的子文档可能有多个属于同一个父文档，会导致返回多个重复的父文档
        所以我们必须对重复的父文档进行去重
        Args:
            需要查找父文档的子文档
        Returns:
            找到的父文档
        '''
        #我们先初步收集父文档，没有排序
        parent_doc_map = {}
        #我们再统计父文档被映射的次数，代表了父文档的重要性
        parent_relevance = {}

        for i, doc in enumerate(child_chunks):
            print(f"chunk {i}: dish={doc.metadata.get('dish_name')}, parent_id={doc.metadata.get('parent_id')}, source={doc.metadata.get('source')}")

        #遍历所有的子文档，统计父文档与相关性次数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                #增加相关性次数
                parent_relevance[parent_id] = parent_relevance.get(parent_id,0)+1

                #缓存父文档
                if parent_id not in parent_doc_map:
                    #检测下这个父文档是否是我们加载的父文档
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_doc_map[parent_id] = doc
                            break

        #根据相关性排序排序,降序,越在前的文档越重要
        sorted_parent_ids = sorted(parent_relevance.keys(),
                            key = lambda x:parent_relevance[x],reverse=True
        )

        #构建最后要返回的父文档
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_doc_map:
                parent_docs.append(parent_doc_map[parent_id])
        
         # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs

        return parent_docs





            









        
    

        


        
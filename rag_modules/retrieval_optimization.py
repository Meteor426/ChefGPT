'''
检索优化模块
'''
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import logging
import traceback

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    '''
    检索优化模块,实现混合检索和过滤
    '''
    def __init__(self,vectorstore:FAISS,chunks:List[Document]) :
        '''
        初始化,并建立两个检索器
        Args:
            vectorstore:FAISS 用于执行稠密向量检索
            chunks:List[Document] 文本块，用于执行稀疏向量检索
        '''
        self.vectorstore = vectorstore
        self.chunks = chunks
        self._set_up_retrievers()
        

    def _set_up_retrievers(self):
        '''
        创建两个检索器
        一个密集向量检索器BM25
        一个稀疏向量检索器
        '''
        try:
            self.vector_retriever = self.vectorstore.as_retriever(
                search_type = "similarity",
                k = 5
            )
        except Exception as e:
            print("=== 捕获异常 ===")
            traceback.print_exc()   # <-- 这个会强制打印完整的traceback
            print("异常类型：", type(e))
            print("异常信息：", e)
        try:
            self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k = 5
            )
        except Exception as e:
            print("=== 捕获异常 ===")
            traceback.print_exc()   # <-- 这个会强制打印完整的traceback
            print("异常类型：", type(e))
            print("异常信息：", e)
        

    def hybrid_search(self,query:str,top_k:int = 3)->List[Document]:
        '''
        混合检索,基于向量检索和BM25检索,使用RRF重排
        Args:
            query:str 查询文本
            top_k:int = 3 返回搜索排名前三的检索文档
        Returns:
            检索到的文本
        '''
        #分别获取两种检索的结果
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        BM25_docs = self.bm25_retriever.get_relevant_documents(query)
        print("🔍 [BM25 检索结果]")
        for i, doc in enumerate(BM25_docs):
            print(f"[{i+1}] 菜名: {doc.metadata.get('dish_name')} | chunk_index: {doc.metadata.get('chunk_index')}")

        # 打印原始 向量 检索结果
        print("🔍 [向量检索结果]")
        for i, doc in enumerate(vector_docs):
            print(f"[{i+1}] 菜名: {doc.metadata.get('dish_name')} | chunk_index: {doc.metadata.get('chunk_index')}")
        #使用rrf重排
        reranked_docs = self._rrf_rerank(vector_docs,BM25_docs)
        return reranked_docs[:top_k]

    def _rrf_rerank(self,vector_docs:List[Document],BM25_docs:List[Document]):
        '''
        RRF重排
        Args:
            vector_docs:向量检索的结果
            BM25_docs:BM25检索的结果
        Returns:
            RRF重排后的文档
        '''
        rrf_scores = {}
        c = 60 #rrf参数
        vec_weight = 1.0
        bm25_weight = 0   
        #计算向量检索的rrf分数
        for rank,doc in enumerate(vector_docs):
            #获取一个id作为k
            doc_id = hash(doc.page_content)
            score = vec_weight * 1 / (rank + c + 1)
            #rrf计算，因为rank是从0开始，所以要加一
            rrf_scores[doc_id] = rrf_scores.get(doc_id,0) + score
        
        #计算BM25检索的rrf分数
        for rank,doc in enumerate(BM25_docs):
            #获取一个id作为k
            doc_id = hash(doc.page_content)
            score = bm25_weight * 1 / (rank + c + 1)
            #rrf计算，因为rank是从0开始，所以要加一
            rrf_scores[doc_id] = rrf_scores.get(doc_id,0) + score

        #合并所有文档并根据rrf分数排序
        #根据id合并 k为hash(doc.page_content)，v为文件本身
        all_docs = {hash(doc.page_content):doc for doc in vector_docs + BM25_docs}
        #排序
        sorted_docs = sorted(all_docs.items(),
                            key=lambda x:rrf_scores.get(x[0],0),   #因为items返回的是元组的集合，{101: "doc1", 102: "doc2"}
                                                                   #所以我们用输入文档x的x[0]来获取键
                            reverse=True
                            )
        #只需要排完序后的doc返回即可
        for i, (doc_id, doc) in enumerate(sorted_docs[:5]):
            print(f"[RRF排序-{i+1}] 菜名：{doc.metadata.get('dish_name', '未知')} | RRF分数：{rrf_scores[doc_id]:.4f}")
        return [doc for _,doc in sorted_docs]
    
    def metadata_filtered_search(self,query:str,filters:Dict[str,Any],
                                top_k:int = 5 
                                )-> List[Document]:
        '''
        带有元数据过滤的搜索
        Args:
            query:查询
            filter:元数据过滤的条件 例如: filters = {"category": "菜谱", "difficulty": ["简单", "中等"]}
            top_k:返回的文档数量
        Returns:
            返回搜索结果
        '''
        #先利用混合检索得到初步结果,这一步把top_k调大，获取更多候选结果用于过滤
        original_docs = self.hybrid_search(query,top_k*3)

        #再进行元数据过滤 这个用法可以固定记住
        filter_docs = []
        for doc in original_docs:    #all函数 判断列表里每一个都为True才返回True，相对的，any函数只要有一个就返回True
            match = all(
                (
                    doc.metadata.get(k) in v if isinstance(v,list)  #如果是一个列表，判断是否在列表内
                    else doc.metadata.get(k) == v   #如果是一个值，判断是否相等
                )
                for k,v in filter.items()   #遍历每一个过滤条件
            )
            if match:
                filter_docs.append(doc)
                if (len(filter_docs) >= top_k):
                    break
        return filter_docs






        
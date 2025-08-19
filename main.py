'''
RAG项目主程序
'''
import os
import sys
import logging
from pathlib import Path
from typing import List

#添加模块搜索路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import RAGConfig, DEFAULT_CONFIG  #刚开始无法识别，添加了一个空的__init__.py文件解决了
from rag_modules import (DataPreparationModule,GenerationIntegrationModule,IndexConstructionModule,RetrievalOptimizationModule)


#加载环境变量
load_dotenv()

#配置日志 设置日志的最低级别、输出格式
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystem:
    '''
    RAG主系统
    '''
    def __init__(self,config:RAGConfig = None):
        '''
        初始化RAG系统
        '''
        self.config = config or DEFAULT_CONFIG  #如果没有config文件就用默认的
        #初始化四个模块
        self.data_module = None
        self.generation_module = None
        self.index_module = None
        self.retrieval_module = None

        #检查下数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")
    
        #检查下API
        if not os.getenv("API_KEY"):
            raise ValueError("请设置API!!")
        
    def initial_system(self):
        '''
        初始化所有模块
        '''
        logger.info("正在加载数据模块.....")
        self.data_module = DataPreparationModule(self.config.data_path)
        
        logger.info("正在加载索引模块.....")
        self.index_module = IndexConstructionModule(self.config.embedding_model,
                                                    self.config.vector_index_path)

        logger.info("正在加载集成模块.....")
        self.generation_module = GenerationIntegrationModule(self.config.model_name,
                                                            self.config.temperature,
                                                            self.config.max_tokens,self.config.history_window_size)
        logger.info("...系统已经加载完成...")

    
    def build_knowledge_base(self):
        '''
        构建索引向量等一系列操作
        '''
        #先尝试加载索引
        vectorstore = self.index_module.load_index()
        
        if vectorstore is not None:

            print("成功加载已保存的向量索引！")

            #由于在检索优化阶段还需要利用到文档

            #加载父文档
            print("正在加载食谱文档")
            documents = self.data_module.load_documents()

            #加载子文档
            print("正在划分子文档")
            chunks = self.data_module.chunk_documents()
        else:

            print("未检测到已有的向量索引，重新建立...")

            #加载父文档
            print("正在加载食谱文档")
            documents = self.data_module.load_documents()

            #加载子文档
            print("正在划分子文档")
            chunks = self.data_module.chunk_documents()

            #构建索引
            print("正在构建索引")
            self.index_module.set_up_embedding()
            vectorstore = self.index_module.build_vector_index(chunks)

            #保存索引
            print("正在保存索引")
            self.index_module.save_index()
        
        print("...正在准备检索优化模块...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore,chunks)
        logger.info("RetrievalOptimizationModule 初始化完成")

    def ask_question(self,question:str,stream:bool = False,history:List[dict] = None):
        '''
        回答用户问题的函数
        Args:
            question:用户的问题
            stream:是否是流式输出
            history:回答上下文
        Returns:
            LLM回答
        '''

        #先检查知识库是否构建完毕
        if not all([self.retrieval_module,self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"用户问题： {question} \n")

        #为xxb添加彩蛋
        if question.strip().lower() == 'kyy':
            print(
            "\n🌸 亲爱的 KYY 🌸\n"
            "在这个知识系统中，有一个彩蛋只为你而生。\n\n"
            "💖 你是特别的、可爱的、值得被世界温柔以待的人。\n"
            "🍽️ 无论你想吃什么，我都会尽力帮你找到最棒的做法。\n"
            "☀️ 愿你每天都有好心情，好胃口，还有一点点小幸运～\n\n"
            "💌 —— 来自你的专属美食 AI\n")
            return None

        #没有历史就初始化
        history = history or []

        #查询路由
        router_type = self.generation_module.query_router(question)
        #如果是闲聊的话
        if router_type == 'chitchat':
            print("闲聊类问题，不触发RAG检索\n")
            return self.generation_module.generate_chitchat_answer(question, history)
        #根据类型来判断是否需要重写query
        if router_type == 'list':
            print("list类型问题，不需要重写\n")
            query = question
        else:
            query = self.generation_module.query_rewrite(question,history)
            print(f"问题已经自动智能重写为:{query}\n")
        
        #利用混合检索进行检索
        relevant_chunks = self.retrieval_module.hybrid_search(query)

        if relevant_chunks:
            print(f"找到了{len(relevant_chunks)}个相关文档\n")
        else:
            print("没有找到相关文档\n")

        for i, doc in enumerate(relevant_chunks):
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            source = doc.metadata.get("source", "未知路径")
            chunk_index = doc.metadata.get("chunk_index", "无")
            chunk_content = doc.page_content.strip().replace('\n', ' ')[:200]  # 只显示前200个字符，避免太长

            print(f"[{i+1}] 菜名：{dish_name} | chunk_index：{chunk_index} | 来源：{source}")
            print(f"    🔍 内容片段：{chunk_content}")
            print("-" * 80)
       
        
        #先找到相关的父文档用于回答问题
        relevant_docs = self.data_module.get_parent_document(relevant_chunks)

        #展示下找到的菜名
        doc_names = []

        print(f"\n🔎 找到了 {len(relevant_docs)} 个相关文档，对应菜品如下：\n")

        for i, doc in enumerate(relevant_docs):
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            source = doc.metadata.get("source", "未知路径")
            chunk_index = doc.metadata.get("chunk_index", "无")
            
            doc_names.append(dish_name)
            
            print(f"[{i+1}] 菜名：{dish_name} | chunk_index：{chunk_index} | 来源：{source}")

        print("\n📋 菜品汇总：")
        print("，".join(doc_names))

        #分类回答问题
        if router_type == 'list':
            #返回回答器
            return self.generation_module.generate_list_answer(query,relevant_docs)
        elif router_type == 'detail':
            return self.generation_module.generate_detail_answer(query,relevant_docs,history)
        else:
            return self.generation_module.generate_general_answer(query,relevant_docs,history)
        
    def run_interactive(self):
        '''
        构建一个交互式系统
        '''
        print("=" * 60)
        print("🍳  欢迎来到 知味小厨 （ChefGPT） 🍳".center(60))
        print("=" * 60)
        print("🤖 基于 RAG 技术的智能菜谱问答系统，助您轻松掌厨！".center(60))
        print("=" * 60)

        #初始化系统
        self.initial_system()
        print("系统初始化完成\n")

        #构建知识库
        self.build_knowledge_base()

        #初始化对话历史
        history = []

        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in [ '退出','exit','quit']:
                    print("👋 感谢使用 ChefGPT，再见！")
                    break

                #获取回答
                answer = self.ask_question(user_input, stream=False,history = history)
                #跳过彩蛋
                if not  answer:
                    continue

                #添加用户信息到历史
                history.append({'role':'user','content':user_input})
                #添加系统消息到历史
                history.append({'role':'assistant','content':answer})

                print(f"{answer}\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}\n")

def main():
    '''
    主函数入口
    '''
    try:
        rag = RAGSystem()
        rag.run_interactive()
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"处理问题时出错: {e}\n")

if __name__ == "__main__":
    main()
    






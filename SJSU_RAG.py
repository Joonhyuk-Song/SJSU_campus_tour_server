import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader



class PDFRAGPipeline:
    def __init__(self, pdf_path: str, model: str = "llama3.2:1b", embed_model: str = "nomic-embed-text"):
        self.pdf_path = pdf_path
        self.model = model
        self.embed_model = embed_model
        self.vector_db = None
        self.llm = ChatOllama(model=self.model)
        self._setup()

    def _setup(self):
        ollama.pull(self.embed_model)

    def ingest_pdf(self):
        if not self.pdf_path:
            raise ValueError("No PDF file provided.")
        loader =  PyPDFLoader(file_path=self.pdf_path)
        data = loader.load()
        return data

    def split_text(self, data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
        chunks = text_splitter.split_documents(data)
        return chunks

    def create_vector_db(self, chunks):
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=self.embed_model),
            collection_name="simple-rag",
        )
    def retrieve_documents(self):
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
    )
        return MultiQueryRetriever.from_llm(self.vector_db.as_retriever(), self.llm, prompt=query_prompt)

    def query(self, question: str):
        if not self.vector_db:
            raise ValueError("Vector database not initialized.")
        
        retriever = self.retrieve_documents()
        rag_prompt = ChatPromptTemplate.from_template("""
        Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | self.llm | StrOutputParser()
        )
        return chain.invoke(input=question)

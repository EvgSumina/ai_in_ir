from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.document_loaders.pdf import PDFMinerLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from pydantic import BaseModel


class UserText(BaseModel):
    message: str
    user_id: str


class TinkoffBase(BaseLoader):
    def __init__(self, file_path="data", chunk_size=1000, chunk_overlap=200):
        self.file_path = file_path
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
            length_function=len,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder="sentences",
            encode_kwargs={"normalize_embeddings": False}
        )

    def load(self):
        csv_load = DirectoryLoader(self.file_path, glob="**/*.csv", loader_cls=CSVLoader)
        csv_chunks = self.text_splitter.split_documents(csv_load.load())
        pdf_load = DirectoryLoader(self.file_path, glob="**/*.pdf", loader_cls=PDFMinerLoader)
        pdf_chunks = self.text_splitter.split_documents(pdf_load.load())
        self.db = Chroma.from_documents(pdf_chunks + csv_chunks, self.embeddings)

    def find_similar(self, q, k=4):
        res = []
        for doc in self.db.similarity_search(q, k):
            res.append(doc.page_content)
        return "\n".join(res)


class ChatBot:
    def __init__(self):
        self.db = TinkoffBase(chunk_size=300, chunk_overlap=50)
        self.db.load()

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path="llama-2-7b-chat.Q5_K_M.gguf",
            use_mlock=True,
            n_ctx=2048,
            last_n_tokens_size=2000,
            callback_manager=callback_manager,
            verbose=True,
            temperature=0,
            n_threads=6,
            model_kwargs={"keep": -1},
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """You are a call center employee at the Russian bank Tinkoff, and you have to answer customer questions.
                    Please respond informatively and respectfully, avoiding any rude or unethical responses. Let's give short but
                    useful answers. If you can't answer the question, please be honest about it.
                    You must answer in russian."""
                ),
                SystemMessagePromptTemplate.from_template(
                    "По этой теме есть следующая информация: {context}"
                ),
                HumanMessagePromptTemplate.from_template(
                    "Представься сотрудником банка. Ответь на этот вопрос на русском языке: {question}?"
                ),
            ]
        )

    def __call__(self, item):
        context = self.db.find_similar(item.message, k=3)
        conversation = LLMChain(llm=self.llm, prompt=self.prompt, verbose=True)
        answer = conversation({"question": item.message, "context": context})
        return answer["text"]

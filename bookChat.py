# Standard library imports
import os
import tempfile

# Third-party imports
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from warnings import simplefilter
from langchain.callbacks import get_openai_callback

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')


def generate_summary(uploaded_file, openai_api_key: str, num_clusters: int = 11, verbose: bool = False) -> str:
    """Generate a summary for a given book."""

    def load_book(file_obj, file_extension):
        """Load the content of a book based on its file type."""
        text = ""
        
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_obj.read())
            
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file.name)
                pages = loader.load()
                for page in pages:
                    text += page.page_content
            elif file_extension == ".epub":
                loader = UnstructuredEPubLoader(temp_file.name)
                data = loader.load()
                text = "\n".join([element.page_content for element in data])
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
            
            os.remove(temp_file.name)  # Clean up the temporary file after use
        
        text = text.replace('\t', ' ')
        return text

    # Get file extension from uploaded file
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    # Load the content of the book from the uploaded file
    text = load_book(uploaded_file, file_extension)
    llm3_turbo = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1000, model='gpt-3.5-turbo-16k')
    num_tokens = llm3_turbo.get_num_tokens(text)
    if verbose:
        print(f"This book has {num_tokens} tokens in it")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
    docs = text_splitter.create_documents([text])
    num_documents = len(docs)
    if verbose:
        print(f"Now our book is split up into {num_documents} documents")

    # Adjust the number of clusters if necessary
    num_clusters = min(num_clusters, num_documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors = embeddings.embed_documents([x.page_content for x in docs])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    #tsne = TSNE(n_components=2, random_state=42)
    #vectors_array = np.array(vectors)
    #reduced_data_tsne = tsne.fit_transform(vectors_array)

    # Uncomment this if you wanna plot the embeddings
    # plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.title('Book Embeddings Clustered')
    # plt.show()

    closest_indices = []
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
    selected_indices = sorted(closest_indices)

    map_prompt = """
    You are provided with a passage from a book. Your task is to produce a comprehensive summary of this passage. Ensure accuracy and avoid adding any interpretations or extra details not present in the original text. The summary should be at least three paragraphs long and fully capture the essence of the passage.
    ```{text}```
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    selected_docs = [docs[doc] for doc in selected_indices]
    summary_list = []

    for i, doc in enumerate(selected_docs):
        current_tokens = llm3_turbo.get_num_tokens(doc.page_content)
        if verbose:
            print(f"Chunk #{i} token count: {current_tokens}")
            print(f"Using llm3_turbo for chunk #{i}")
        map_chain = load_summarize_chain(llm=llm3_turbo, chain_type="stuff", prompt=map_prompt_template)
        chunk_summary = map_chain.run([doc])
        summary_list.append(chunk_summary)
        if verbose:
            print(f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")
    summaries = "\n".join(summary_list)
    summaries = Document(page_content=summaries)
    if verbose:
        print(f"Your total summary has {llm3_turbo.get_num_tokens(summaries.page_content)} tokens")

    llm4 = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=3000, model='gpt-4', request_timeout=120)

    combine_prompt = """
    You are presented with a series of summarized sections from a book. Your task is to weave these summaries into a single, cohesive, and verbose summary. The reader should be able to understand the main events or points of the book from your summary. Ensure you retain the accuracy of the content and present it in a clear and engaging manner.
    ```{text}```
    COHESIVE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    with get_openai_callback() as cost:
        reduce_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=combine_prompt_template)
        output = reduce_chain.run([summaries])
        if verbose:
            print(output)
            print(cost)

    return output

# testing
if __name__ == '__main__':
    book_path = "../Happy_Sexy_Millionaire.epub"
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    # Mimic Streamlit's uploaded file behavior using open
    with open(book_path, 'rb') as uploaded_file:
        summary = generate_summary(uploaded_file, openai_api_key, verbose=True)
        print(summary)


#cost with gpt 3.5 4k context: Total Cost (USD): $0.12513
#cost with gpt 3.5 16k context: Total Cost (USD): $0.13191
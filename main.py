""" Awesome Knowledge Challenge """
import jsonlines
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Annoy
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.evaluation import load_evaluator

import os
import dotenv


dotenv.load_dotenv()

OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))

# Define the Silarity upper limit for updated contents
SIMILARITY_LIMIT = 1.0

# Define a template for API call
TEMPLATE = """
Based on given Information below, the respective knowledge in the Text below should be updated.
Answer would be the updated version of the text.
Please, do not change the irrelevant knowledge in the text.

Text: {text}
Information: {information}

Answer: Updated version of the Text according to the Information.
"""


def read_jsonl_file(file_path: str) -> list[dict[str, str]]:
    """
    Read JSONL file and collect all texts along with their URLs
    
    Args:
        file_path: path of jsonl file

    Return:
        return the list of lines containing content and url
    """
    lines = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            lines.append({'content': obj['content'], 'url': obj['url']})
    return lines


def write_jsonl_file(file_path: str, lines: list[dict[str, str]]) -> None:
    """
    Write the updated documents back to a new JSONL file

    Args:
        file_path: path to write the jsonl file
        lines: content of the jsonl file

    Return:
        None
    """
    # Write the updated documents back to a new JSONL file
    with jsonlines.open(file_path, mode='w') as writer:
        for line in lines:
            writer.write({'content': line['content'], 'url': line['url']})


def update_document_content(similar_pages: tuple[Annoy, float], 
                            lines: list[dict[str, str]], 
                            llm_chain: LLMChain, 
                            query: str) -> dict[str, str]:
    """
    Update documents based on similarity search results.
    
    Args:
        similar_pages: List of tuples containing a dictionary for the page content and a similarity score.
        lines: List of dictionaries containing the content of the JSONL input file.
        llm_chain: LLM chain used to call the LLM API
        query: natural language query to automatically update the documentation
    Returns:
        return the previous contents with updated contents to evaluate it.
    """
    updated_contents = {}
    for page in similar_pages:
        # Limit the similarity score of the page to filter out less relevant results.
        if page[1] > SIMILARITY_LIMIT:
            break
        # Get response according to the similar content and save them with the updated content
        current_content = page[0].page_content
        response = llm_chain.invoke({"text": current_content, "information": query})
        # Update the changed contents in the lines and print them before and after the update
        for line in lines:
            if line['content']== current_content:
                print("before: " , line['content'])
                line['content']= response["text"]
                print("after: " , line['content'])
                updated_contents[current_content] = response["text"]
    return updated_contents

def string_distance_evaluation(updated_contents: dict[str, str]):
    """
    Check the edit distance between old content and updated content to 
    see whether it is changed or not. 
    Print the edit distance for each updated content

    Args:
        updated_contents: dictionary containing the previous contents 
            with updated contents
    """
    evaluator = load_evaluator("string_distance")
    for i, old_content in enumerate(updated_contents):
        string_distance = evaluator.evaluate_strings(
            prediction=updated_contents[old_content],
            reference=old_content,
        )
        print(f"string_distance for similar content {i}: ", string_distance)

def semantic_distance_evaluation(embeddings_func: HuggingFaceEmbeddings, 
                                 updated_contents: dict[str, str]):
    """
    Use Embedding Distance to compare the smeantic similarity of old content and 
    updated content with the query. According to my hypothesis updated content
    should be more similar to the query since it is updated according to it.
    Print the similarity distance  values for each updated content

    Args:
        embeddings_func: Embedding model
        updated_contents: dictionary containing the previous contents 
            with updated contents
    """
    hf_evaluator = load_evaluator("embedding_distance", embeddings=embeddings_func)
    # semantic similarity
    for i, old_content in enumerate(updated_contents):
        previous_semantic_similarity_to_query = hf_evaluator.evaluate_strings(
            prediction=old_content,
            reference=query,
        )
        new_semantic_similarity_to_query = hf_evaluator.evaluate_strings(
            prediction=updated_contents[old_content],
            reference=query,
        )
        print(f"Semantic Similarity of Old Content with Query for {i}:", previous_semantic_similarity_to_query)
        print("Semantic Similarity of Updated Content with Query for {i}:", new_semantic_similarity_to_query)


def main(query: str):
    """
    Main function for Awesome Knowledge Challenge. It uses the following steps:

    1-) Load scraped data
    2-) Use each line as a seperate chunk
    3-) Use the content of the line to generate embeddings with free Hugging Face model
    4-) Store them as vector database with Annoy
    5-) Query vector store with given query
    6-) Take most similar 50 pages
    7-) Update them with generic Prompt template with OpenAI GPT-3.5-turbo model
        if similarity socore is below the threshold
    8-) Print the previous page content and update page content
    9-) Check the Edit distance between old and updated content
    10-) Check the semantic embedding similarities between old content and updated
        content with the given natural language query
    11-) Write the updated content in another Jsonl file.

    Args:
        query: natural language query to automatically update the documentation 
    """

    # Setup for Hugging Face embeddings and Annoy vector store
    embeddings_func = HuggingFaceEmbeddings()

    # Read JSONL file and collect all texts along with their URLs
    lines = read_jsonl_file('scraped_data.jsonl')

    # Only use content of the page in the vector database
    texts = [line['content'] for line in lines]

    # Store them as vector database
    vector_store = Annoy.from_texts(texts, embeddings_func)

    # Query to check similarity of contents, limit it to 50
    similar_pages = vector_store.similarity_search_with_score(query, k=50)

    # Use gpt-3.5-turbo since performance is not critical
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", api_key=OPENAI_API_KEY)

    # Create an LLMChain instance with prompt template and LLM model
    prompt_template = PromptTemplate.from_template(TEMPLATE)
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    # Iterate over similar pages, update each corresponding content, and collect results
    updated_contents = update_document_content(similar_pages, lines, llm_chain, query)

    # Compare the Edit distance between old content and updated content to check if it is updated
    string_distance_evaluation(updated_contents)
    # Compare the semantic similarity of old content and updated content with given query
    semantic_distance_evaluation(embeddings_func, updated_contents)

    # Write the updated version
    write_jsonl_file('updated_scraped_data.jsonl', lines)


if __name__ == "__main__":
    query = (
    "We removed the ability to archive queries, and instead "
    "added the ability to completely delete them. Update all "
    "relevant knowledge"
    )
    main(query)

# Awesome Knowledge Challenge:

Given by AwesomeQA

## Minimum requirements:**

- Based on a generic query like the above, the respective knowledge should be updated.
- The number of OpenAI requests made per query should be limited to 50.
- After a query was executed successfully, the updated json should be stored and the changes should be displayed / printed.
- There are no performance or speed requirements.
- Quickly evaluate the results, either quantitatively or qualitatively.
- LLMs are still limited, so it’s ok if it doesn’t always work perfectly. Focus on the approach instead.


## Structure:

It solves the problem with the following steps:

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


## Improvements:

1-) Annoy is read-only - once the index is built you cannot add any more embeddings!
Thus, different solutions might be used and embeddings can be updated on the fly.

2-) More advanced models might be used for Embedding generation.

3-) Solution might be tested with different use cases. 
        Extreme cases:
        - Query which is irrelavent to all page contents
        - Query which is highly relevant to all page contents

4-) Clustering similar page content and storing them like that might be tried.

5-) Test functions should be written.

6-) Prompt template can be improved.

7-) Better evaluation techniques might be used.
# Awesome Knowledge Challenge:

Provided by AwesomeQA. 
This project tackles the challenge of maintaining up-to-date documentation in dynamically changing environments. The approach focuses on automation, leveraging AI to streamline the update process.

## Minimum requirements:**

- Based on a generic query like the above, the respective knowledge should be updated.
- The number of OpenAI requests made per query should be limited to 50.
- After a query was executed successfully, the updated json should be stored and the changes should be displayed / printed.
- There are no performance or speed requirements.
- Quickly evaluate the results, either quantitatively or qualitatively.
- LLMs are still limited, so it’s ok if it doesn’t always work perfectly. Focus on the approach instead.


## Structure:

It solves the problem with the following steps:

    1-) Load scraped data.
    2-) Treat each line as a seperate chunk
    3-) Generate embeddings for each line using a free Hugging Face model.
    4-) Store embeddings in an Annoy vector store library.
    5-) Query the vector store using the specified query.
    6-) Retrieve the 50 most similar pages.
    7-) Update the pages using a generic prompt template with the OpenAI GPT-3.5-turbo model 
    if the similarity score is below the threshold.
    8-) Print the content of the page before and after updates.
    9-) Evaluate the edit distance between the original and updated content.
    10-) Assess the semantic embedding similarities between the old and updated content relative to the query.
    11-) Write the updated content into another JSONL file.


## Improvements:

1-) Annoy is read-only - once the index is built you cannot add any more embeddings!
Thus, different solutions like pure vector databases might be used and embeddings can be updated on the fly.
Also, experimenting with other vector libraries like FAISS might offer better performance.

2-) More advanced models might give better performance for Embedding generation.

3-) The solution should be tested across various scenarios, including:
        - Query which is irrelavent to all page contents
        - Query which is highly relevant to all page contents

4-) Clustering similar page contents and storing them together might increase the performance.

5-) Test functions should be written.

6-) System prompt template can be improved.

7-) Evaluation is done qualitatively and with simple quantitative analysis. 
Better evaluation techniques might be used to check the content pre- and post-update to ensure that 
the changes align with the natural language query.

8-) Similarity threshold to update is chosen by qualitative checks and it might be improved.
Also, secondary LLM might be used to confirm if the content really needs updating based on the change context.
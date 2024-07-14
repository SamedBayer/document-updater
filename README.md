# Updating Knowledge Challenge:

Provided by AwesomeQA. 
This project tackles the challenge of maintaining up-to-date documentation in dynamically changing environments. The approach focuses on automation, leveraging AI to streamline the update process.

## How to Run:

- Install poetry [Poetry Page](https://python-poetry.org/docs/)
- Run the following code:
 `poetry install`

- Add the OpenAI API key to the .env file.
- Run the main function:
  `poetry run python3 main.py`

- You can also change the given query in the main calling function in the lowest part of the main.py.


## Structure:

It solves the problem with the following steps:

    1-) Load scraped data.
    2-) Treat each line as a separate chunk
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


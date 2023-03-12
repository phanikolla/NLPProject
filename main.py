
import os
from os.path import dirname, join, realpath
import uvicorn
import openai
from fastapi import FastAPI

app = FastAPI(
    title="NLP SQL API",
    description="A simple API that use NLP model to predict the SQL",
    version="0.1",
)

# load the sentiment model
#
openai.api_key = "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"
def generate_sql_query(prompt, max_tokens=1024, temperature=0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    response = openai.Completion.create(
        model="curie:ft-personal:customercomplaint-26-02-2023-2023-02-26-07-27-52",
        prompt=prompt,
        max_tokens=1024,
        temperature=temperature,
        top_p=top_p,
        n=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)
    return response.choices[0].text


def query_database(prompt):
    # Set up the prompt
    # prompt = f"Query the database:\n{prompt}\nResponse:"
    request = f"### Postgres SQL tables, with their properties:\n#\n# Consumercomplaints(Complaint_ID, Company, product_name , subproduct name ,Date_received)\n#\n{prompt}\n "

    # Call the Curie model to generate the response
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=request,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated response from the API response
    generated_text = response.choices[0].text.strip()

    return generated_text



@app.get("/get-sql")
def getSQL(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    #cleaned_review = text_leaning(review)
    print(review)
    # perform prediction
    #prediction = model.predict([cleaned_review])
    #output = int(prediction[0])
    #probas = model.predict_proba([cleaned_review])
    #output_probability = "{:.2f}".format(float(probas[:, output]))

    # output dictionary
    #sentiments = {0: "Negative", 1: "Positive"}
    # sql = generate_sql_query(review)
    # print(sql)
    sql1 = query_database(review)
    print(sql1)
    # show results
    result = {"sql": sql1, "sql1": sql1}
    return sql1

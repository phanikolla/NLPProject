
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
        #engine="davinci-codex",
        engine="text-davinci-002",
        #engine="davinci-instruct-beta-v3",
        #model="curie:ft-personal:customercomplaint-10-03-2023-2023-03-10-06-47-35",
        ##prompt="### Postgres SQL tables, with their properties:\n#\n# Consumercomplaints(Complaint_ID, Company, Product,Date_received)\n#\n### A query to list the companies and the count of complaints from each company\n",
        ##prompt="### Postgres SQL tables, with their properties:\n#\n# Consumercomplaints(Complaint_ID, Company, product_key,Date_received,State,Submitted_via)\n# product_mstr(product_key,product_name,product_desc,create_date,updated_date)\n# sub_prod_mstr(subproduct_key,subproduct_name,subproduct_desc,created_dt,updated_dt)\n# prod_mapping(product_key,subproduct_key,mapping_id,created_dt,updated_dt) \n#\n##"+ prompt +"\n#",
        prompt="### Postgres SQL tables, with their properties:\n#\n# Consumercomplaints(Complaint_ID, Company, prod_subprod_mapping_key ,Date_received,State,Submitted_via,Timely_response)\n# Product_master(product_ID,product_name,product_desc,created_date,updated_date)\n# sub_prod_master(subproduct_ID,subproduct_name,subproduct_desc,created_dt,updated_dt)\n# prod_mapping(product_ID,subproduct_ID,prod_subprod_mapping_key,created_dt,updated_dt) \n#\n##"+ prompt +"\n#",
        ##prompt="### Postgres SQL tables, with their properties:\n#\n# Consumercomplaints(Date_received,mapping_id,Issue,Sub_issue,Consumer_complaint_narrative,Company_public_response,Company,State,ZIP_code,Tags,Consumer_consent_provided,Submitted_via,Date_sent_to_company,Company_response_to_consumer,Timely_response,Consumer_disputed,Complaint_ID)\n# Product_master(product_ID,product_name,product_desc,created_date,updated_date)\n# sub_prod_master(subproduct_ID,subproduct_name,subproduct_desc,created_date,updated_dt)\n# prod_mapping(product_ID,subproduct_ID,mapping_id,created_date,updated_date) \n#\n### Return total number of complaints for Consumer Loan product?",
        ##prompt="### Postgres SQL tables, with their properties:\n#\n# Consumercomplaints(Complaint_ID, Company, product,subproduct,Date_received,State,Submitted_via,Timely_response)"+ prompt +"\n#",
      
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=1,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=[";"]
    )
    return response.choices[0].text.strip()


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

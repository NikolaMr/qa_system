# QA system
This repo is a showcase for Haystack QA skills.

## Requirements
In order to make this work, make sure you have Python 3.8+ available.

The recommended way for starting is to:

1. create a virtual environment by invoking `python -m venv venv`
2. activating it by invoking `source venv/bin/activate` 
3. installing requirements by invoking `pip install -r requirements.txt`.

## Starting the server
To start the server, invoke `uvicorn qa_system:app --reload`. `--reload` flag is here in case you want live changes to take effect. If not, feel free to drop that flag.

### Getting answers
After starting the server you can ask it some questions like
```text
who is the father of Arya Stark
``` 
by issuing a GET call to http://127.0.0.1:8000/answers?who+is+father+of+Arya+Stark and you'll get the answer.

### Adding new documents
You can add new documents by issuing a POST to http://127.0.0.1:8000/documents with json payload like this:
```json
{
  "text": "Hiking is a long, vigorous walk, usually on trails or footpaths in the countryside. Walking for pleasure developed in Europe during the eighteenth century. Religious pilgrimages have existed much longer but they involve walking long distances for a spiritual purpose associated with specific religions."
}
```
After the document was successfully indexed, you can ask a question like `what is hiking` and the server will be able to respond to your query.
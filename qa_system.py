import logging
import os
import tempfile

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import fetch_archive_from_http
from haystack.nodes import TextConverter, PreProcessor
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack import Pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Text

app = FastAPI()
indexing_pipeline = None
querying_pipeline = None


class Answer(BaseModel):
    text: Text


@app.on_event('startup')
def startup():
    init_haystack()


def init_haystack():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.DEBUG)


    # Get the host where Elasticsearch is running, default to localhost
    host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

    document_store = ElasticsearchDocumentStore(
        host=host,
        username="",
        password="",
        index="document"
    )

    document_store.delete_documents(index="document")

    doc_dir = "data/build_a_scalable_question_answering_system"

    fetch_archive_from_http(
        url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt3.zip",
        output_dir=doc_dir
    )

    global indexing_pipeline
    indexing_pipeline = Pipeline()
    text_converter = TextConverter()
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_header_footer=True,
        clean_empty_lines=True,
        split_by="word",
        split_length=200,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )

    indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
    indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

    files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
    indexing_pipeline.run_batch(file_paths=files_to_index)

    retriever = BM25Retriever(document_store=document_store)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    global querying_pipeline
    querying_pipeline = Pipeline()
    querying_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    querying_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])


def run_query(querying_pipeline, query):
    prediction = querying_pipeline.run(
        query=query,
        params={
            "Retriever": {"top_k": 10},
            "Reader": {"top_k": 5}
        }
    )

    return prediction


@app.get('/answers')
def get_answers(query: str):
    global querying_pipeline
    return run_query(querying_pipeline, query)


@app.post('/answers')
def index_answer(answer: Answer):
    tf = tempfile.NamedTemporaryFile('w')
    name = tf.name
    tf.write(answer.text)
    tf.seek(0)
    indexing_pipeline.run_batch(file_paths=[name])
    tf.close()
    return 'success'

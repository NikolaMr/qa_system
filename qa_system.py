import logging
import os
import tempfile

from haystack.document_stores import FAISSDocumentStore, BaseDocumentStore
from haystack.utils import fetch_archive_from_http
from haystack.nodes import TextConverter, PreProcessor
from haystack.nodes import EmbeddingRetriever, BaseRetriever
from haystack.nodes import FARMReader
from haystack import Pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Text, Optional
import json

app = FastAPI()
indexing_pipeline: Optional[Pipeline] = None
querying_pipeline: Optional[Pipeline] = None
document_store: Optional[BaseDocumentStore] = None
retriever: Optional[BaseRetriever] = None


class Document(BaseModel):
    text: Text


@app.on_event('startup')
def startup():
    init_haystack()


def init_haystack():
    logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.DEBUG)

    with open('config.json', 'r') as f:
        config = json.load(f)

    global document_store
    if os.path.exists(config['faiss_index_path']):
        document_store = FAISSDocumentStore.load(config['faiss_index_path'])
    else:
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    global indexing_pipeline
    indexing_pipeline = Pipeline()
    text_converter = TextConverter()


    if config['preprocessor']['split_by'] == 'sentence':
        preprocessor = PreProcessor(
            clean_whitespace=True,
            clean_header_footer=True,
            clean_empty_lines=True,
            split_by="sentence",
            split_length=10,
            split_overlap=2,
            split_respect_sentence_boundary=False
        )
    else:
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

    global retriever
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=config['retriever']['embedding_model'],
        model_format="sentence_transformers",
        use_gpu=True
    )

    if config['requires_indexing']:
        document_store.delete_documents()

        doc_dir = "data/build_a_scalable_question_answering_system"

        fetch_archive_from_http(
            url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt3.zip",
            output_dir=doc_dir
        )

        files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)][:10]
        indexing_pipeline.run_batch(file_paths=files_to_index)
        document_store.update_embeddings(retriever, update_existing_embeddings=False)

        if config['faiss_index_path']:
            document_store.save(config['faiss_index_path'])

    reader = FARMReader(model_name_or_path=config['reader']['model_name_or_path'], use_gpu=True)

    global querying_pipeline
    querying_pipeline = Pipeline()
    querying_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    querying_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])


def run_query(querying_pipeline: Pipeline, query: str):
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


@app.post('/documents')
def index_document(document: Document):
    tf = tempfile.NamedTemporaryFile('w')
    name = tf.name
    tf.write(document.text)
    tf.seek(0)
    indexing_pipeline.run_batch(file_paths=[name])
    tf.close()
    try:
        document_store.update_embeddings(retriever, update_existing_embeddings=False)
        with open('config.json', 'r') as f:
            config = json.load(f)
        if config['faiss_index_path']:
            document_store.save(config['faiss_index_path'])
    except AttributeError:
        logging.log(logging.INFO, 'Can\'t update embeddings, document store doesn\'t support it.')
    return 'success'

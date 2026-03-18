from builder import Builder
from chunker import Chunker
from ner import NER


class Indexing:
    def __init__(self, builder: Builder, chunker: Chunker, ner: NER):
        self.builder = builder
        self.chunker = chunker
        self.ner = ner

    def run(self):
        
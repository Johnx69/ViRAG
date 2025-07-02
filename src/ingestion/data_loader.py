from datasets import load_dataset
from llama_index.core import Document
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class VietnameseWikipediaLoader:
    def __init__(self, subset: str = "20231101.vi"):
        self.subset = subset

    def load_data(self, num_samples: Optional[int] = None) -> List[Document]:
        try:
            logger.info(f"Loading Vietnamese Wikipedia dataset: {self.subset}")
            dataset = load_dataset("wikimedia/wikipedia", self.subset)

            if num_samples:
                dataset = dataset["train"].select(
                    range(min(num_samples, len(dataset["train"])))
                )
            else:
                dataset = dataset["train"]

            documents = []
            for idx, item in enumerate(dataset):
                doc = Document(
                    text=item["text"],
                    metadata={
                        "title": item["title"],
                        "url": item["url"],
                        "doc_id": item["id"],
                        "source": "vietnamese_wikipedia",
                    },
                )
                documents.append(doc)

                if idx % 1000 == 0:
                    logger.info(f"Processed {idx} documents")

            logger.info(f"Loaded {len(documents)} documents from Vietnamese Wikipedia")
            return documents

        except Exception as e:
            logger.error(f"Error loading Wikipedia data: {e}")
            raise

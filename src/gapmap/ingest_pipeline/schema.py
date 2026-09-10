from __future__ import annotations


from pydantic import BaseModel, ConfigDict, Field, field_validator
import pymupdf
from typing import Any, List, Dict
import torch
import numpy as np


strict_config = ConfigDict(extra='forbid', strict=True, arbitrary_types_allowed=True)


class RawResultData(BaseModel):
    model_config = strict_config

    scores: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor
    polygon_points: List[np.ndarray]
    order_seq: torch.Tensor

    def iter_elements(self):
        s = self.scores.tolist()
        l = self.labels.tolist()
        b = self.boxes.tolist()
        o = self.order_seq.tolist()

        for i in range(len(s)):
            yield (s[i], l[i], b[i], self.polygon_points[i], o[i])


class PageLayoutData(BaseModel):
    model_config = strict_config

    page_obj: pymupdf.Page
    page_pixmap: pymupdf.Pixmap
    raw_results: List[RawResultData] = Field(default_factory=list)


class DocumentSection(BaseModel):
    model_config = strict_config

    text: str = Field(...)
    tables: Dict[str, str] = Field(default_factory=dict)
    images: Dict[str, str] = Field(default_factory=dict)


class CleanedDocument(BaseModel):
    model_config = strict_config

    title: str = Field(...)
    sections: List[DocumentSection] = Field(default_factory=list)
    id: str = Field(...)
    authors: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    abstract: str = Field(...)
    updated: str = Field(...)
    published: str = Field(...)

# result schema using open ragbench layout
# https://github.com/vectara/open-rag-bench
# {
#     "title": "Paper Title",
#     "sections": [
#         {
#             "text": "Section text content with placeholders for tables/images",
#             "tables": {"table_id1": "markdown_table_string", ...},
#             "images": {"image_id1": "base64_encoded_string", ...},
#         },
#         ...
#     ],
#     "id": "Paper ID",
#     "authors": ["Author1", "Author2", ...],
#     "categories": ["Category1", "Category2", ...],
#     "abstract": "Abstract text",
#     "updated": "Updated date",
#     "published": "Published date"
# }
from __future__ import annotations


from pathlib import Path
import pymupdf
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from gapmap.utils.config import settings
from gapmap.ingest_pipeline.schema import PageLayoutData
from gapmap.ingest_pipeline.core.result_handler import process_results


device = settings.device
model_path = settings.models_dir / "PP_DocLayoutV3_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, local_files_only=True).to(device)
image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)


def process_doc(file_path: Path):
    with pymupdf.open(file_path) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=96)
            page_image = pix.pil_image()
            inputs = image_processor(images=page_image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = image_processor.post_process_object_detection(outputs, target_sizes=[page_image.size[::-1]])
            layout = PageLayoutData(
                page_obj=page,
                page_pixmap=pix,
                raw_results=results,
            )

            process_results(layout)



for file_path in settings.ingest_dir.glob("**/*.pdf"):
    process_doc(file_path)





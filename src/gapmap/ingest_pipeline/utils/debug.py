from __future__ import annotations


from transformers.models.pp_doclayout_v3.modeling_pp_doclayout_v3 import PPDocLayoutV3ForObjectDetection
from gapmap.ingest_pipeline.utils.scale import scale_bbox
from matplotlib import pyplot as plt
from matplotlib import patches
from gapmap.ingest_pipeline.schema import PageLayoutData


def debug_draw_bbox(layout_data: PageLayoutData, model: PPDocLayoutV3ForObjectDetection) -> None:
    page_image = layout_data.page_pixmap.pil_image()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(page_image)
    for result in layout_data.raw_results:
        for idx, (score, label_id, box, polygon_points, _) in enumerate(result.iter_elements()):
            box = tuple(round(i, 2) for i in box)
            result_label = model.config.id2label[label_id]
            x0, y0, x1, y1 = box

            width = x1 - x0
            height = y1 - y0

            rect = patches.Rectangle(
                (x0, y0),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )

            ax.add_patch(rect)
            ax.text(x0, y0 - 5, f"{str(idx)} {result_label}", color='blue', fontsize=10)
    ax.axis('off')
    plt.show()


def debug_pdf_box(layout_data: PageLayoutData) -> None:
    page = layout_data.page_obj
    for result in layout_data.raw_results:
        for idx, (score, label_id, box, polygon_points, _) in enumerate(result.iter_elements()):
            box = tuple(round(i, 2) for i in box)
            box = scale_bbox(layout_data.page_pixmap, layout_data.page_obj, box)
            shape = page.new_shape()
            shape.draw_rect(box)
            shape.finish(
                width=2.0,
                color=(1, 0, 0),
            )
            shape.commit()
    pix = page.get_pixmap(dpi=96)
    page_image = pix.pil_image()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(page_image)
    ax.axis('off')
    plt.show()
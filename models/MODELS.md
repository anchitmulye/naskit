# h3dnas Models
 
| Model | Dataset | Base Acc | NAS Acc | Δ | Params ↓ | Speedup | Base ONNX | H3DNAS ONNX |
|---|---|---|---|---|---|---|---|---|
| PointNet | ModelNet40 | 90.32% | 90.44% | +0.22pp | 32.2% | 1.50x | [pointnet_cls_c40_n1024.onnx](../models/pointnet/pointnet_cls_c40_n1024.onnx) | [pointnet_cls_c40_n1024_h3dnas.onnx](../models/pointnet/pointnet_cls_c40_n1024_h3dnas.onnx) |
| PointNet++ | ModelNet40  | 91.90% | 91.61% | -0.28pp | 5.7% | 3.28x | [pointnet2_cls_ssg_c40_n1024.onnx](../models/pointnet2/pointnet2_cls_ssg_c40_n1024.onnx) | [pointnet2_cls_ssg_c40_n1024_h3dnas.onnx](../models/pointnet2/pointnet2_cls_ssg_c40_n1024_h3dnas.onnx) |
<!-- | PointMLP | ModelNet40 (2468 test) | 92.95% | xx.xx% | +x.xxpp | 14.4% | | [base](models/pointmlp/base.onnx) | [h3dnas](models/pointmlp/nas_finetuned.onnx) |
| PTv3 | ModelNet10 (908 test) | 11.23% | — | — | — | | [base](models/ptv3/base.onnx) | [h3dnas]() | -->
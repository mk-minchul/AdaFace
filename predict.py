import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

import net
from face_alignment import mtcnn


def detect_faces(mtcnn_model, path, limit_faces=None):
    img = Image.open(path).convert('RGB')
    boxes, faces = mtcnn_model.align_multi(img, limit=limit_faces)

    faces = np.stack([np.array(face) for face in faces])
    faces = ((faces[..., ::-1] / 255.) - 0.5) / 0.5
    faces = faces.transpose(0,3,1,2)
    faces = torch.from_numpy(faces).float()
    return np.array(boxes), faces


def load_pretrained_model(model_name):
    NAME_2_ARCH = {'adaface_ir18_vgg2': 'ir_18', 'adaface_ir50_ms1mv2': 'ir_50', 'adaface_ir101_webface12m': 'ir_101'}
    model = net.build_model(NAME_2_ARCH[model_name])
    statedict = torch.load(f"pretrained/{model_name}.ckpt")['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


def box2cvbox(box):
    ing_box = box.astype(int)
    p1 = ing_box[0], ing_box[1]
    p2 = ing_box[2], ing_box[3]
    return p1, p2


def score_to_color(score):
    return [int((1-score)*255), 0 ,int(score*255)]


class Predictor(BasePredictor):
    def setup(self):
        self.mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))

    def predict(self,
                query_image: Path = Input(description="Image with a face to search"),
                reference_image: Path = Input(description="Image with multiple faces to search in"),
                model_name: str = Input(description="Name of the architecture and dataset",
                                        default='adaface_ir50_ms1mv2',
                                        choices=['adaface_ir50_ms1mv2', 'adaface_ir18_vgg2', 'adaface_ir101_webface12m']),
                ) -> Path:
        query_image = str(query_image)
        reference_image = str(reference_image)
        model_name = str(model_name)

        # Load model
        model = load_pretrained_model(model_name)

        # Extract and align faces
        q_boxes, q_faces = detect_faces(self.mtcnn_model, query_image, limit_faces=1)
        r_boxes, r_faces = detect_faces(self.mtcnn_model, reference_image)

        # Extract feature vectors
        q_fv, _ = model(q_faces)
        r_fvs = model(r_faces)

        # Compute Cosine similarities between normalized vectors
        similarities = (q_fv @ r_fvs[0].T)[0].tolist()

        # Print Bboxs and similarity scores on image
        img = cv2.imread(reference_image)
        for i in range(len(r_boxes)):
            color = score_to_color(similarities[i])
            p1, p2 = box2cvbox(r_boxes[i])
            cv2.rectangle(img, p1, p2, color=color, thickness=2)
            p1 = p1[0] - 5, p1[1] - 5
            text = f"{similarities[i]:.3f}"
            cv2.putText(img, text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=color, thickness=1, lineType=cv2.LINE_AA)

        out_path = "output.png"
        cv2.imwrite(out_path, img)

        return Path(out_path)

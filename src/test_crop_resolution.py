import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from PIL import Image
from prw_evaluation import _load_query_crop, embed_detected_crops


class CropResolutionTest(unittest.TestCase):
    def test_query_and_gallery_use_saved_resolution(self):
        class Model:
            reid_head = SimpleNamespace(crop_size=(256,128))
            def embed_crop(self, crops):
                self.crops = crops.clone()
                return torch.ones(len(crops),4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'image.png'
            Image.new('RGB',(180,300),color=(130,45,20)).save(path)
            model=Model()
            embed_detected_crops(model,path,torch.tensor([[0.,0.,180.,300.]]),'cpu')
            query=_load_query_crop(path, model.reid_head.crop_size)
            self.assertEqual(tuple(model.crops.shape),(1,3,256,128))
            self.assertTrue(torch.equal(model.crops[0],query))
            self.assertEqual(tuple(_load_query_crop(path).shape),(3,128,64))

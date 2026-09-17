import unittest
import torch
from dinov3_reid import DinoV3CropReID
from dino_reid import crop_epoch
from metric_loss import crop_to_crop_reid_loss, SupervisedNTXentLoss
from training import make_grad_scaler
from test_dinov3_reid import TinyBackbone

class ClassificationTests(unittest.TestCase):
    def test_combined_objective_mapping_update_and_restore(self):
        torch.manual_seed(31)
        model = DinoV3CropReID(backbone=TinyBackbone(), embedding_dim=4, classifier_ids=[90,7]).eval()
        targets = [dict(ids=torch.tensor([7,90]), image_id=torch.tensor(i),
                        query_crops=torch.randn(2,3,32,16)) for i in range(2)]
        loader=[(None,targets)]
        crops=torch.cat([t['query_crops'] for t in targets])
        with torch.no_grad():
            emb,logits=model(crops,return_logits=True)
            metric,_=crop_to_crop_reid_loss(emb,torch.tensor([7,90,7,90]),
                torch.tensor([0,0,1,1]),SupervisedNTXentLoss(.1))
            ce=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,0,1]))
        result=crop_epoch(model,loader,torch.device('cpu'),metric_loss='supcon',classification_weight=.5)
        self.assertAlmostEqual(result['total_loss'],float(metric+.5*ce),places=6)
        self.assertEqual(result['classification_known_crops'],4)
        before=model.identity_classifier.weight.detach().clone()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        crop_epoch(model,loader,torch.device('cpu'),opt,make_grad_scaler(torch.device('cpu')),
                   metric_loss='supcon',classification_weight=1.)
        self.assertFalse(torch.equal(before,model.identity_classifier.weight))
        self.assertIsNotNone(model.embedding_proj.weight.grad)
        self.assertIsNone(model.backbone.conv.weight.grad)
        clone=DinoV3CropReID(backbone=TinyBackbone(),embedding_dim=4,classifier_ids=[7,90]).eval()
        clone.load_state_dict(model.state_dict(),strict=True)
        model.eval()
        self.assertTrue(torch.equal(model(crops),clone(crops)))
        self.assertEqual(tuple(model(crops).shape),(4,4))

    def test_validation_unknown_ids_not_mapped_to_training_classes(self):
        model=DinoV3CropReID(backbone=TinyBackbone(),classifier_ids=[7,90])
        targets=[dict(ids=torch.tensor([999]),image_id=torch.tensor(i),
                      query_crops=torch.randn(1,3,32,16)) for i in range(2)]
        loader=[(None,targets)]
        result=crop_epoch(model,loader,torch.device('cpu'),metric_loss='supcon',classification_weight=1.)
        self.assertEqual(result['classification_loss'],0.)
        self.assertEqual(result['classification_known_crops'],0)
        opt=torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError,'missing'):
            crop_epoch(model,loader,torch.device('cpu'),opt,make_grad_scaler(torch.device('cpu')),
                       metric_loss='supcon',classification_weight=1.)

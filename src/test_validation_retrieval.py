import unittest
import torch
from validation_retrieval import rank_validation

class ValidationRankingTest(unittest.TestCase):
    def test_source_exclusion_duplicates_and_misses(self):
        box = torch.tensor([[0.,0.,10.,10.]])
        gallery = [dict(gt=box, ids=torch.tensor([1]), boxes=box),
                   dict(gt=box, ids=torch.tensor([1]), boxes=box.repeat(2,1)),
                   dict(gt=box, ids=torch.tensor([1]), boxes=torch.empty(0,4))]
        result = rank_validation(torch.tensor([[1.,0.],[.9,0.],[.8,0.]]),
                                 [torch.tensor([1.,0.])],gallery,[(0,0)])
        self.assertEqual(result['retrieval_mAP'], .5)
        self.assertEqual(result['retrieval_top1'], 1.)
        for g in gallery:g['boxes']=torch.empty(0,4)
        empty=rank_validation(torch.empty(0,2),[torch.tensor([1.,0.])],gallery,[(0,0)])
        self.assertEqual(empty['retrieval_mAP'],0.)
        self.assertEqual(empty['retrieval_queries'],1)

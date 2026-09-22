import unittest
from unittest.mock import patch
import torch
from final_comparison import state_fingerprint, fixed_occurrences, plot_similarities, load_system

class FinalComparisonTests(unittest.TestCase):
    def test_fingerprint_independent_of_dictionary_order(self):
        a={'b':torch.tensor(2.),'a':torch.ones(2)}
        self.assertEqual(state_fingerprint(a),state_fingerprint(dict(reversed(list(a.items())))))
        self.assertNotEqual(state_fingerprint(a),state_fingerprint({'b':torch.tensor(3.),'a':torch.ones(2)}))

    def test_sampling_distinct_frames_and_deterministic(self):
        with patch('scipy.io.loadmat',return_value={'img_index_test':None}), \
             patch('prw_dataset._mat_strings',return_value=['a','b','c']), \
             patch('final_comparison._load_gallery_target',return_value=(torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]]),torch.tensor([7,8]))):
            a=fixed_occurrences('.',num_ids=2,per_id=2,seed=4)
            b=fixed_occurrences('.',num_ids=2,per_id=2,seed=4)
        self.assertEqual(a,b)
        for identity in (7,8):self.assertEqual(len({r['frame'] for r in a if r['identity']==identity}),2)

    def test_similarity_masks_same_frame_not_only_diagonal(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        records=[dict(identity=1,frame='a'),dict(identity=1,frame='b'),dict(identity=2,frame='a')]
        banks={'test':dict(query=torch.eye(3),gallery=torch.eye(3))}
        fig=plot_similarities(banks,records)
        mask=fig.axes[0].images[0].get_array().mask
        self.assertTrue(mask[0,2]);self.assertFalse(mask[0,1]);self.assertTrue(mask[1,1])
        plt.close(fig)

    def test_wrong_strategy_fails_before_model_construction(self):
        with patch('torch.load',return_value={'strategy':'crop_to_crop'}):
            with self.assertRaisesRegex(ValueError,'strategy'):
                load_system(dict(kind='swin_scene',checkpoint_path='fake'),None,'cpu')

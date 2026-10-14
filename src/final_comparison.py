"""Evaluation-only loading and matched visualizations for the final PRW report."""
from collections import defaultdict
from pathlib import Path
import hashlib
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from model import PersonSearchModel
from dino_reid import DinoPersonSearch
from dinov3_reid import DinoV3CropReID
from swin_reid import SwinSceneReID, SwinCropReID
from prw_evaluation import (_image_tensor, _load_gallery_target, embed_detected_crops)


def state_fingerprint(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def resolve_checkpoint(spec, directory):
    """Explicit version and file; optionally verify it was produced by expected run."""
    if spec.get('local_path'):
        path=Path(spec['local_path'])
    else:
        import wandb
        artifact=wandb.Api().artifact(spec['artifact'],type='model')
        if spec.get('run_id'):
            entity,project=spec['artifact'].split('/')[:2]
            run=wandb.Api().run(f"{entity}/{project}/{spec['run_id']}")
            if artifact.id not in {a.id for a in run.logged_artifacts()}:
                raise ValueError(f"Artifact does not belong to run {spec['run_id']}")
        path=Path(artifact.download(root=str(Path(directory)/spec['artifact'].replace('/','__').replace(':','-'))))/spec['filename']
    if not path.is_file():raise FileNotFoundError(path)
    return path


def load_system(spec, detector_path, device):
    """Loads user-selected latest/best weights exactly; never silently substitutes."""
    state=torch.load(spec['checkpoint_path'],map_location='cpu',weights_only=False)
    kind=spec['kind']; scene=kind in ('swin_scene','resnet_scene')
    if kind.startswith('swin_'):
        strategy='crop_to_scene' if scene else 'crop_to_crop'
        if state.get('strategy')!=strategy:raise ValueError('Swin checkpoint strategy mismatch')
        config=dict(state['detector_config']);config.update(pretrained=False,detection_only=True)
        detector=PersonSearchModel(**config)
        crop_size=tuple(state['run_config']['crop_size'])
        if scene:
            dim=state['model_state_dict']['reid_head.embedding_proj.weight'].shape[0]
            model=SwinSceneReID(detector,embedding_dim=dim,crop_size=crop_size)
        else:
            dim=state['model_state_dict']['reid_head.embedding_proj.weight'].shape[0]
            model=DinoPersonSearch(detector,SwinCropReID(pretrained=False,embedding_dim=dim,crop_size=crop_size))
        model.load_state_dict(state['model_state_dict'],strict=True)
        image_size=tuple(state['run_config']['image_size'])
    elif kind=='dino':
        det=torch.load(detector_path,map_location='cpu',weights_only=False)
        config=dict(det['model_config']);config.update(pretrained=False,detection_only=True)
        detector=PersonSearchModel(**config)
        detector.load_state_dict(det['model_state_dict'],strict=True)
        rc=dict(state['reid_config']);rc['pretrained']=False
        # Optional local copy of the DINOv3 code (no Internet needed); weights come from the checkpoint.
        if spec.get('dinov3_repo'):rc['hub_repo']=str(spec['dinov3_repo'])
        model=DinoPersonSearch(detector,DinoV3CropReID(**rc))
        model.reid_head.load_state_dict(state['reid_state_dict'],strict=True)
        image_size=tuple(det.get('detector_image_size',det['run_config']['image_size']))
    elif kind in ('resnet_scene','resnet_crop'):
        config=dict(state['model_config'])
        if not config.get('backbone_name','resnet50').startswith('resnet'):raise ValueError('Not a ResNet checkpoint')
        config.update(pretrained=False,detection_only=True)
        detector=PersonSearchModel(**config)
        detector.load_state_dict(state['model_state_dict'],strict=True)
        if spec.get('expected_metric_loss') and config.get('metric_loss') != spec['expected_metric_loss']:
            raise ValueError('ResNet checkpoint loss does not match the supplied experiment')
        if kind == 'resnet_scene':
            model=SwinSceneReID(detector,embedding_dim=config.get('embedding_dim',256),
                                crop_size=tuple(spec.get('crop_size',(128,64))))
            model.reid_head.load_state_dict(detector.reid_head.state_dict(),strict=True)
        else:
            # Original shared crop encoder: do not replace its trained ReID head.
            model=detector
            model.reid_head.crop_size=tuple(spec.get('crop_size',(128,64)))
        image_size=tuple(state['run_config']['image_size'])
    else:raise ValueError('Unknown system kind')
    detector_state=(model.detector if hasattr(model,'detector') else model).state_dict()
    # ReID head stored in PersonSearchModel is unused by all frozen detectors.
    fingerprint=state_fingerprint({k:v for k,v in detector_state.items() if not k.startswith('reid_head.')})
    provenance=dict(epoch=state.get('epoch'),strategy=state.get('strategy'),
                    run_config=state.get('run_config',{}),checkpoint_path=spec['checkpoint_path'],
                    artifact=spec.get('artifact'),selection=spec.get('selection'),
                    detector_sha256=fingerprint,detector_image_size=image_size,
                    gallery_embedding_mode='scene' if scene else 'crop',
                    detector_config=config,
                    total_parameters=sum(p.numel() for p in model.parameters()))
    return model.to(device).eval().requires_grad_(False),provenance


def fixed_occurrences(root,num_ids=20,per_id=8,seed=0):
    """Choose labeled GT boxes from official test frames; one occurrence per frame."""
    from scipy.io import loadmat
    from prw_dataset import _mat_strings
    root=Path(root)
    names=_mat_strings(loadmat(root/'frame_test.mat')['img_index_test'])
    by_id=defaultdict(dict)
    for name in names:
        boxes,ids=_load_gallery_target(root,name)
        for box,identity in zip(boxes,ids.tolist()):
            if identity>=0 and bool((box[2:]>box[:2]).all()):
                by_id[identity].setdefault(name,box.tolist())
    eligible=sorted(i for i,items in by_id.items() if len(items)>=per_id)
    if len(eligible)<num_ids:raise ValueError('Not enough identities for requested fixed sample')
    rng=np.random.default_rng(seed)
    chosen=sorted(rng.choice(eligible,num_ids,replace=False).tolist())
    records=[]
    for identity in chosen:
        frames=sorted(by_id[identity])
        selected=sorted(rng.choice(len(frames),per_id,replace=False).tolist())
        records.extend(dict(identity=identity,frame=frames[j],box=by_id[identity][frames[j]]) for j in selected)
    return records


@torch.inference_mode()
def occurrence_embeddings(model,root,records,device,image_size,mode):
    """Same GT pixels for all systems; query and gallery paths are both preserved."""
    query=[];gallery=[]
    for r in records:
        boxes=torch.tensor([r['box']],dtype=torch.float32)
        crop=embed_detected_crops(model,Path(root)/'frames'/f"{r['frame']}.jpg",boxes,device)
        query.append(crop)
        if mode=='scene':
            image,scale=_image_tensor(Path(root),r['frame'],size=image_size)
            gallery.append(model.embed_scene(image[None].to(device),[(boxes*scale).to(device)]).cpu())
        else:gallery.append(crop)
    return dict(query=F.normalize(torch.cat(query).float(),dim=1),
                gallery=F.normalize(torch.cat(gallery).float(),dim=1))


def plot_embeddings(banks,records,seed=0):
    """Per-model t-SNE, fixed sample/hyperparameters. Axes are not shared geometry."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    identities=sorted({r['identity'] for r in records})
    labels=np.array([identities.index(r['identity']) for r in records])
    fig,axes=plt.subplots(1,len(banks),figsize=(6*len(banks),5),squeeze=False)
    for ax,(name,bank) in zip(axes[0],banks.items()):
        features=bank['gallery'].numpy()
        points=TSNE(n_components=2,perplexity=min(30.,(len(features)-1)/3),
                    init='pca',learning_rate='auto',random_state=seed).fit_transform(features)
        ax.scatter(points[:,0],points[:,1],c=labels,cmap='tab20',vmin=0,vmax=19,s=16)
        ax.set_title(name);ax.set_xticks([]);ax.set_yticks([])
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker='o',linestyle='',color=plt.get_cmap('tab20')(i/19),label=str(identity)) for i,identity in enumerate(identities)]
    fig.legend(handles=handles,title='Identity',loc='lower center',ncol=10,bbox_to_anchor=(.5,-.07))
    fig.suptitle('GT-gallery embeddings: same identities/occurrences; independent t-SNE projections')
    fig.tight_layout()
    return fig


def plot_similarities(banks,records):
    import matplotlib.pyplot as plt
    ids=np.array([r['identity'] for r in records]);frames=np.array([r['frame'] for r in records])
    mask=frames[:,None]==frames[None,:]
    boundaries=np.flatnonzero(ids[1:]!=ids[:-1])+1
    fig,axes=plt.subplots(1,len(banks),figsize=(6*len(banks),5),squeeze=False)
    for ax,(name,bank) in zip(axes[0],banks.items()):
        sim=(bank['query']@bank['gallery'].T).numpy()
        sim=np.ma.array(sim,mask=mask)
        im=ax.imshow(sim,vmin=-1,vmax=1,cmap='coolwarm')
        for b in boundaries:ax.axhline(b-.5,color='black',lw=.4);ax.axvline(b-.5,color='black',lw=.4)
        ax.set_title(name);ax.set_xlabel('Gallery occurrences grouped by ID');ax.set_ylabel('Query occurrences grouped by ID')
        fig.colorbar(im,ax=ax,fraction=.046)
    fig.suptitle('Cosine similarity; same-frame entries masked, identical color scale')
    fig.tight_layout()
    return fig


def plot_matched_query(results,root,query_key,top_k=5):
    """One model per row, with the same query and ranked crops in every row."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(results),top_k+1,figsize=(3*(top_k+1),3.5*len(results)),squeeze=False)
    for row,(name,records) in enumerate(results.items()):
        record=next(r for r in records if (r['identity'],r['frame'])==query_key)
        for ax in axes[row]:ax.axis('off')
        with Image.open(record['query_path']) as image:axes[row,0].imshow(image.convert('RGB'))
        axes[row,0].set_title(f"{name}\nQuery {query_key[0]}")
        for col,item in enumerate(record['ranked'][:top_k],1):
            with Image.open(Path(root)/'frames'/f"{item['frame']}.jpg") as image:
                axes[row,col].imshow(image.convert('RGB').crop(tuple(item['box'])))
            axes[row,col].set_title(f"Rank {col}: {'correct' if item['correct'] else 'wrong'}\nsim={item['score']:.3f}",
                                   color='forestgreen' if item['correct'] else 'crimson')
    fig.suptitle(f'Same query across systems — source frame {query_key[1]}')
    fig.tight_layout()
    return fig

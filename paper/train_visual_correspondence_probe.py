"""Fixed paired-crop visual-adaptation development comparison."""

import argparse
import json
from pathlib import Path

from paper.decode_movi_geometry_pilot import digest

LABELS = ('close', 'far', 'occluded')


def load_split(corpus, partition):
    if partition not in ('train', 'validation'):
        raise ValueError('only development train/validation allowed')
    manifest=json.loads((corpus/'manifest.json').read_text())
    source=corpus/f'{partition}.jsonl'
    if digest(source)!=manifest['exports_sha256'][partition]:
        raise ValueError('export hash mismatch')
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    expected={r['id']:r for r in manifest['records'] if r['partition']==partition and r['status']=='labeled'}
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=set(expected):
        raise ValueError('full split required')
    for row in rows:
        entry=expected[row['id']]
        path=(corpus/row['image']).resolve()
        if (not path.is_relative_to(corpus.resolve()) or row['image']!=entry['image']
                or row['target']!=entry['label'] or row['target'] not in LABELS
                or digest(path)!=entry['image_sha256']):
            raise ValueError('image/target provenance mismatch')
    return rows


def run(corpus, output):
    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torchvision.models import resnet18

    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',scope='small development visual-adaptation diagnostic, not VLM comparison, camera accuracy, or DNF',
        protocol='first and second native 161x161 panel crops; ImageNet ResNet18; frozen backbone versus trainable layer4; same seed-0 head and training order; 20 epochs; Adam 1e-4; batch 8; inverse-frequency training loss; no augmentation, early stopping or checkpoint selection; validation only after both training arms finish',
        limitation='188 training examples and two validation videos; cropped evidence omits context; pretrained generic backbone differs from Qwen vision encoder; no isolated claim about cause of VLM failure',
        corpus_sha256=digest(corpus/'manifest.json'), script_sha256=digest(Path(__file__)),arms={})
    def save():
        (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.manual_seed(0)
        train_rows=load_split(corpus,'train')
        def tensors(rows):
            images=[]
            for row in rows:
                with Image.open(corpus/row['image']) as im:
                    a=np.asarray(im.convert('RGB')).copy()
                assert a.shape==(383,544,3)
                images.append(np.stack([a[214:375,51:212],a[214:375,331:492]]))
            x=torch.tensor(np.stack(images)).permute(0,1,4,2,3).float()/255
            x=(x-torch.tensor([.485,.456,.406])[None,None,:,None,None])/torch.tensor([.229,.224,.225])[None,None,:,None,None]
            y=torch.tensor([LABELS.index(r['target']) for r in rows])
            return x,y
        x,y=tensors(train_rows)
        url='https://download.pytorch.org/models/resnet18-f37072fd.pth'
        weights=torch.hub.load_state_dict_from_url(url,model_dir=str(output/'weights'),check_hash=True,map_location='cpu')
        report.update(pretrained_url=url,pretrained_sha256=digest(output/'weights/resnet18-f37072fd.pth'),
            train_ids=[r['id'] for r in train_rows],train_class_counts=torch.bincount(y,minlength=3).tolist(),
            torch=torch.__version__,gpu=torch.cuda.get_device_name(0))
        class Pair(nn.Module):
            def __init__(self,adapt):
                super().__init__()
                self.backbone=resnet18(weights=None); self.backbone.load_state_dict(weights)
                self.backbone.fc=nn.Identity()
                for p in self.backbone.parameters(): p.requires_grad=False
                if adapt:
                    for p in self.backbone.layer4.parameters(): p.requires_grad=True
                self.head=nn.Sequential(nn.Linear(2048,128),nn.ReLU(),nn.Linear(128,3))
            def forward(self,batch):
                f=self.backbone(batch.flatten(0,1)).reshape(-1,2,512)
                a,b=f[:,0],f[:,1]
                return self.head(torch.cat([a,b,(a-b).abs(),a*b],dim=1))
        class_weights=(len(y)/(3*torch.bincount(y,minlength=3).float())).cuda()
        for arm in ('frozen','adapted'):
            torch.manual_seed(0)
            model=Pair(arm=='adapted').cuda()
            # Frozen normalization statistics in both arms isolate weight updates.
            model.eval()
            parameters=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(parameters,lr=1e-4)
            record=dict(status='training',trainable_parameters=sum(p.numel() for p in parameters),epochs=[])
            report['arms'][arm]=record; report['status']='training'; save()
            for epoch in range(20):
                order=torch.randperm(len(y),generator=torch.Generator().manual_seed(epoch))
                total=0.
                for indices in order.split(8):
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(x[indices].cuda())
                    loss=nn.functional.cross_entropy(logits,y[indices].cuda(),weight=class_weights)
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                    loss.backward()
                    norm=nn.utils.clip_grad_norm_(parameters,1.)
                    if not torch.isfinite(norm): raise FloatingPointError('nonfinite gradient')
                    optimizer.step(); total+=float(loss.detach())*len(indices)
                record['epochs'].append(dict(epoch=epoch+1,mean_batch_weighted_loss=total/len(y)));save()
                print(arm,epoch+1,total/len(y),flush=True)
            torch.save(model.state_dict(),output/f'{arm}.pt')
            record.update(status='trained',checkpoint_sha256=digest(output/f'{arm}.pt'));save()
            del model,optimizer; torch.cuda.empty_cache()
        # No validation images or labels are loaded until both arms are trained.
        validation_rows=load_split(corpus,'validation')
        vx,vy=tensors(validation_rows)
        assert not ({r['id'] for r in train_rows}&{r['id'] for r in validation_rows})
        report['status']='evaluating';save()
        for arm in ('frozen','adapted'):
            model=Pair(False).cuda()
            model.load_state_dict(torch.load(output/f'{arm}.pt',map_location='cuda',weights_only=True));model.eval()
            predictions=[]
            with torch.inference_mode():
                for batch in vx.split(8): predictions.extend(model(batch.cuda()).softmax(-1).cpu().tolist())
            report['arms'][arm].update(status='complete',validation=[dict(id=r['id'],target=r['target'],probabilities=p,predicted=LABELS[int(np.argmax(p))]) for r,p in zip(validation_rows,predictions,strict=True)])
            save();del model;torch.cuda.empty_cache()
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally: save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();run(args.corpus,args.output)

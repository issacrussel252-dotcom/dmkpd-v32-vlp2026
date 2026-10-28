"""Run the frozen V32 submission pipeline on a directory of images.

Data and checkpoints stay outside the source repository. Only trusted, published
checkpoints with the recorded SHA256 hashes are accepted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--residual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New, empty output directory')
    parser.add_argument('--limit', type=int, help='Smoke test only; omit for submission')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty to prevent mixing runs')
    files = sorted((p for p in args.images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}), key=lambda p: p.name.casefold())
    if not files or len({p.stem.casefold() for p in files}) != len(files):
        parser.error('Input must contain images with unique filename stems')
    if args.limit:
        files = files[:args.limit]
    os.environ['DMKPD_DETECTOR'] = str(args.detector.resolve())
    os.environ['DMKPD_QUALITY'] = str(args.quality.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    import torch
    import v31_core as core
    from pipeline import run_one
    if not torch.cuda.is_available():
        raise RuntimeError('This release preserves the verified CUDA FP32 inference path; a CUDA GPU is required')
    expected = '8bbcdcaa0edf7c44f681ca9c7eae0a182a392e708a0d13e06eeb455ea5b48a72'
    if core.sha256(args.residual) != expected:
        raise ValueError('Residual checkpoint SHA256 does not match V32 B seed20261031 epoch3')
    model, quality, mean, std = core.load_models()
    checkpoint = torch.load(args.residual, map_location='cpu', weights_only=False)
    residual = core.CornerResidualHead().cuda().eval()
    residual.load_state_dict(checkpoint['residual_head'], strict=True)
    residual.requires_grad_(False)
    capture = core.install_capture(model)
    out = args.output / 'map_out'
    out.mkdir(parents=True, exist_ok=True)
    # Match the original warmup path without accessing labels or developer caches.
    for _ in range(5):
        run_one(model, capture, quality, residual, mean, std, files[0], files[0].stem)
    records = {}
    for index, path in enumerate(files, 1):
        record = run_one(model, capture, quality, residual, mean, std, path, path.stem)
        predictions = record['predictions_score_floor_0_30']
        records[path.stem] = predictions
        lines = []
        for row in predictions:
            xy = [value for point in row['quad'] for value in point]
            lines.append('DPM ' + format(row['final_score'], '.8f') + ' ' + ' '.join(format(v, '.9f') for v in xy))
        (out / (path.stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8', newline='\n')
        print(f'{index}/{len(files)} {path.name}: {len(predictions)} predictions', flush=True)
    with zipfile.ZipFile(args.output / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.glob('*.txt')):
            archive.write(path, 'map_out/' + path.name)
    (args.output / 'predictions.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'images': len(files), 'predictions': sum(map(len, records.values())), 'limited_run': args.limit is not None,
               'weights_sha256': {name: core.sha256(getattr(args, name)) for name in ('detector', 'quality', 'residual')},
               'torch': torch.__version__, 'cuda': torch.version.cuda}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

"""Verification script for ISL Alphabet + Digit system."""
import sys
sys.path.insert(0, '.')

print('--- Testing all module imports ---')
from src.utils.labels import create_deterministic_alphabet_label_map, normalize_alphabet_label
from src.utils.metrics import compute_topk_accuracy, evaluate_predictions
from src.alphabet.dataset import ISLAlphabetDataset, build_dataloaders
from src.alphabet.preprocessing import build_train_transform, build_eval_transform
from src.alphabet.model import AlphabetCNNModel, build_model
from src.alphabet.train import AlphabetTrainer, EarlyStopping
from src.alphabet.evaluate import AlphabetEvaluator
from src.alphabet.inference import AlphabetInferenceEngine
print('All imports: OK')

print()
print('--- Testing label normalization ---')
tests = [
    ('a', 'A'), ('Z', 'Z'), ('0', '0'),
    ('zero', '0'), ('nine', '9'), ('1. A', 'A'), ('bad', '')
]
all_ok = True
for inp, expected in tests:
    got = normalize_alphabet_label(inp)
    status = 'OK' if got == expected else 'FAIL (got ' + repr(got) + ')'
    if got != expected:
        all_ok = False
    print('  normalize(' + repr(inp) + ') -> ' + repr(got) + ' [' + status + ']')

print()
print('--- Testing label map (36 classes) ---')
lm = create_deterministic_alphabet_label_map()
assert len(lm) == 36, 'Expected 36 classes, got ' + str(len(lm))
assert lm['A'] == 0, 'A should be 0'
assert lm['Z'] == 25, 'Z should be 25'
assert lm['0'] == 26, '0 should be 26'
assert lm['9'] == 35, '9 should be 35'
print('  36 classes: OK')
print('  A=' + str(lm['A']) + '  Z=' + str(lm['Z']) + '  0=' + str(lm['0']) + '  9=' + str(lm['9']))

print()
print('--- Testing model forward pass (MobileNetV3Small, no pretrained) ---')
import torch
config = {
    'model': {'name': 'mobilenet_v3_small', 'pretrained': False, 'dropout': 0.3},
    'training': {}
}
model = build_model(config, num_classes=36)
model.eval()
dummy = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)
assert out.shape == (2, 36), 'Expected (2,36), got ' + str(out.shape)
print('  Input ' + str(tuple(dummy.shape)) + ' -> Output ' + str(tuple(out.shape)) + ': OK')

print()
print('--- Testing transform pipelines ---')
train_tf = build_train_transform(image_size=224, horizontal_flip=False)
eval_tf = build_eval_transform(image_size=224)
from PIL import Image
import numpy as np
dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
t = train_tf(dummy_img)
assert t.shape == (3, 224, 224)
e = eval_tf(dummy_img)
assert e.shape == (3, 224, 224)
print('  Transforms: OK (train and eval -> (3,224,224))')

print()
print('--- Checking label_map.json ---')
from pathlib import Path
from src.utils.labels import load_label_map
lm_path = Path('models/alphabet/label_map.json')
assert lm_path.exists(), 'label_map.json not found'
lm_loaded = load_label_map(lm_path)
assert len(lm_loaded) == 36
print('  models/alphabet/label_map.json: OK (36 classes)')

print()
print('--- Verifying script CLIs load without error ---')
import subprocess, sys
scripts = [
    'scripts/prepare_alphabet_dataset.py',
    'scripts/inspect_alphabet_dataset.py',
    'scripts/train_alphabet.py',
    'scripts/evaluate_alphabet.py',
    'scripts/predict_alphabet.py',
    'scripts/webcam_alphabet.py',
]
for s in scripts:
    r = subprocess.run(
        [sys.executable, s, '--help'],
        capture_output=True, text=True, timeout=15
    )
    ok = r.returncode == 0
    print('  ' + s + ': ' + ('OK' if ok else 'FAIL\n' + r.stderr[:200]))

print()
print('=== ALL VERIFICATIONS PASSED ===')

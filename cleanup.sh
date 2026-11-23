#!/usr/bin/env bash
# Run from the repository root AFTER copying the new files over the repository.
# Removes obsolete / generated files from Git. `--cached` keeps the local copies on your disk.
set -e

# Obsolete: old alphabet, checkpoint migration, the 131-class checkpoint (retraining is planned anyway)
git rm -f latvian_htr/old_alphabet.py latvian_htr/migrate_checkpoint.py latvian_htr/checkpoints_v2/crnn_best.pt

# Generated / per-user files (now in .gitignore); kept on disk, removed from Git only
git rm -r --cached --ignore-unmatch \
  latvian_htr/__pycache__ \
  latvian_htr/samples \
  latvian_htr/data/real_dataset/labels.csv.bak \
  latvian_htr/data/real_dataset/images/.verify_progress \
  latvian_htr/data/real_dataset/images/.verify_settings.json \
  latvian_htr/data/real_dataset/images/_originals

git status --short | head -50
echo "Check the list above, then: git commit -m 'Reorganise the NN part: alphabet, fallback fonts, real validation split, cleanup'"

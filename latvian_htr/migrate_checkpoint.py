"""
Pārnes vecā čekpointa svarus uz jauno alfabētu, lai nevajadzētu mācīt tīklu no nulles.

Kāpēc tas vajadzīgs: pēdējais slānis (rnn.fc) ir tik izvadu, cik klašu alfabētā. Ja pēc apmācības alfabētam
pievienoti/noņemti simboli, čekpoints neiederas ("size mismatch"). Turklāt CHARS ir sakārtots, tāpēc jauns
simbols pa vidu nobīda visu pārējo indeksus - tīkls "runātu" nepareizus burtus.

Šis skripts: simboliem, kas bija abos alfabētos, pārkopē to rindu (pareizajā vietā); jaunajiem simboliem
iedod mazu sākuma vērtību, lai tie netiktu prognozēti pirms apmācības; CNN un BiLSTM svari paliek kā bija.

Kā skriet (old_alphabet.py = alfabēta fails TĀ, kāds tas bija, kad mācīji čekpointu):
    py migrate_checkpoint.py --old-alphabet old_alphabet.py --checkpoint checkpoints_v2\\crnn_best.pt --out checkpoints_v2\\crnn_best_migrated.pt
Pēc tam apmāci tālāk (iesaka mazāku lr, piem. 3e-4):
    py train.py --resume checkpoints_v2\\crnn_best_migrated.pt --lr 3e-4 --checkpoint-dir checkpoints_v3
"""

import argparse
import importlib.util

import numpy as np

FC_W, FC_B = "rnn.fc.weight", "rnn.fc.bias"


def remap_rows(old_alphabet, new_alphabet, weight, bias, seed=0):
    """Pure numpy. weight [len(old), D], bias [len(old)] -> weight [len(new), D], bias [len(new)] + report."""
    if weight.shape[0] != len(old_alphabet) or bias.shape[0] != len(old_alphabet):
        raise ValueError(f"old alphabet has {len(old_alphabet)} classes, but the checkpoint has {weight.shape[0]}")
    rng = np.random.default_rng(seed)
    old_index = {ch: i for i, ch in enumerate(old_alphabet)}
    new_w = rng.normal(0.0, float(weight.std()), size=(len(new_alphabet), weight.shape[1])).astype(weight.dtype)
    new_b = np.full(len(new_alphabet), float(bias.min()), dtype=bias.dtype)   # unseen symbols start 'unlikely'
    kept, added = 0, []
    for i, ch in enumerate(new_alphabet):
        if ch in old_index:
            new_w[i], new_b[i] = weight[old_index[ch]], bias[old_index[ch]]
            kept += 1
        else:
            added.append(ch)
    dropped = [ch for ch in old_alphabet if ch not in set(new_alphabet)]
    return new_w, new_b, {"kept": kept, "added": added, "dropped": dropped}


def find_state(obj):
    """The dict that holds 'rnn.fc.weight' (a raw state_dict or one nested inside a training checkpoint)."""
    if isinstance(obj, dict):
        if FC_W in obj:
            return obj
        for v in obj.values():
            if isinstance(v, dict):
                found = find_state(v)
                if found is not None:
                    return found
    return None


def load_alphabet_file(path):
    spec = importlib.util.spec_from_file_location("old_alphabet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.ALPHABET)


def main():
    import torch
    from alphabet import ALPHABET as NEW_ALPHABET

    p = argparse.ArgumentParser()
    p.add_argument("--old-alphabet", required=True, help="alphabet.py as it was when the checkpoint was trained")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except Exception:                       # newer torch refuses non-tensor objects by default
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = find_state(ckpt)
    if state is None:
        raise SystemExit(f"'{FC_W}' not found in the checkpoint - is it a CRNN checkpoint?")

    old_alphabet = load_alphabet_file(args.old_alphabet)
    w, b = state[FC_W].numpy(), state[FC_B].numpy()
    new_w, new_b, rep = remap_rows(old_alphabet, NEW_ALPHABET, w, b)
    state[FC_W], state[FC_B] = torch.from_numpy(new_w), torch.from_numpy(new_b)

    if isinstance(ckpt, dict):              # optimizer / scheduler tensors have the old size - drop them
        for key in ("optimizer", "optimizer_state_dict", "scheduler", "scheduler_state_dict"):
            if key in ckpt:
                del ckpt[key]
                print(f"removed '{key}' from the checkpoint (it had the old shape)")
    torch.save(ckpt, args.out)
    print(f"old classes {len(old_alphabet)} -> new classes {len(NEW_ALPHABET)}")
    print(f"copied rows: {rep['kept']} | new symbols (fresh): {rep['added']} | dropped symbols: {rep['dropped']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

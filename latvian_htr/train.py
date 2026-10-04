"""
Neironu tīkla apmācības skripts.

Kā skriet:
    python train.py --steps 2000                     # tikai ar sintētiskajiem datiem
    python train.py --real-data path/to/real_dataset  # pievienot īstos skenējumus, kad tie būs pieejami

Kas ir "apmācība"? Mēs atkārtojam vienu un to pašu ciklu daudzas reizes:
1) tīklam rādām attēlu ar tekstu;
2) tīkls mēģina to nolasīt un pieļauj kļūdu;
3) aprēķina, cik liela ir kļūda (šo skaitli sauc par zudumiem);
4) nedaudz pielāgojam tīkla iekšējos skaitļus (tā "svarus"), lai nākamreiz kļūda būtu mazāka.
Pēc tūkstošiem šādu atkārtojumu tīkls sāk lasīt pareizi.
"""

import argparse   # komandrindas iestatījumu nolasīšana (--steps 2000 u. tml.)
import os         # darbs ar mapēm
import random     # nejauši skaitļi

import torch
from torch.utils.data import ConcatDataset, DataLoader  # datu kopu apvienošana un pakešu ielādētājs

from alphabet import NUM_CLASSES, encode
from dataset import RealHTRDataset, SyntheticHTRDataset, collate_batch
from model import CRNN

torch.set_num_threads(os.cpu_count())

def build_dataset(real_data_path: str | None, synth_length: int):
    """Apkopo datu "noliktavu" apmācībai: mākslīgi piemēri + reāli piemēri."""

    datasets = [SyntheticHTRDataset(length=synth_length)]
    if real_data_path:
        # Ja norādīta mape ar īstajiem skenējumiem, pievienojam arī tos.
        datasets.append(RealHTRDataset(real_data_path))
    # Ja datu kopa ir viena, atgriežam to; ja ir divas, apvienojam vienā (ConcatDataset).
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def train(args):
    """Galvenā apmācības funkcija. args ir lietotāja nodotie iestatījumi."""
    # Izvēlamies, ar ko veikt aprēķinus: videokarte (cuda) ir desmitiem reižu ātrāka par procesoru (cpu).
    # Ja videokarte ir pieejama, izmantojam to, pretējā gadījumā procesoru.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculating on {device}")

    dataset = build_dataset(args.real_data, args.synth_per_epoch)
    # DataLoader - «izsniedzējs»: paņem piemērus no datu kopas un izsniedz tos paketēs.
    # batch_size - cik piemēru ir vienā paketē; shuffle=True - sajaukt
    # secību (lai tīkls neiemācītos secību); collate_fn - kā
    # apvienot piemērus paketē (izlīdzināt pēc platuma); num_workers - cik
    # paralēli palīgi sagatavo datus.
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=args.num_workers,
    )

    # Izveidojam pašu tīklu un pārvietojam to uz izvēlēto ierīci (.to(device)).
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    # Optimizators - «labotājs»: tas zina, kā mainīt tīkla svarus, lai
    # kļūda samazinātos. Adam ir populārs un uzticams variants.
    # lr (learning rate) - «mācīšanās ātrums»: cik lieliem soļiem mainīt svarus.
    # Pārāk lieli soļi - tīkls «lēkā» un nekonverģē, pārāk mazi - tas mācās bezgalīgi.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # CTCLoss - funkcija, kas aprēķina kļūdu mainīga garuma tekstam
    # (blank=0 - «tukšā» simbola numurs). zero_infinity=True neļauj apmācībai
    # sabojāties, ja kādam piemēram kļūda pēkšņi kļūst bezgalīga.
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    # Mape saglabātajiem failiem. exist_ok=True: ja mape jau pastāv, viss ir kārtībā.
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0          # apmācības soļu skaitītājs (viens solis = viena piemēru pakete)
    model.train()     # ieslēdzam «apmācības režīmu» (daži slāņi tajā darbojas citādi nekā novērtēšanas režīmā)
    while step < args.steps:
        # Viens gājiens caur datu kopu. Katra cikla iterācija ir viena pakete:
        # attēli, to teksti un attēlu platumi.
        for images, texts, widths in loader:
            images = images.to(device)  # pārvietojam attēlus turpat, kur atrodas tīkls

            # Sagatavojam CTC «pareizās atbildes». Katru tekstu pārveidojam par
            # simbolu numuriem un apvienojam tos vienā garā sarakstā, bet
            # target_lengths saglabājam, cik simbolu ir katrā tekstā.
            targets, target_lengths = [], []
            for t in texts:
                enc = encode(t)
                targets.extend(enc)
                target_lengths.append(len(enc))
            targets = torch.tensor(targets, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)

            # Tiešā pāreja: tīkls aplūko attēlus un izvada prognozes.
            log_probs = model(images)  # [T, B, C] - T soļi, B attēli, C simboli
            T = log_probs.size(0)      # maksimālais soļu skaits paketē (pēc platākā attēla)

            # SVARĪGI: attēli labajā pusē tika papildināti līdz kopējam platumam max_w.
            # widths satur katra attēla patieso platumu PIRMS padding.
            # Tāpēc katram piemēram atsevišķi aprēķinām, cik soļu
            # faktiski iegūts pēc CNN. Daļa pēc šī soļa
            # attiecas tikai uz padding un nedrīkst piedalīties CTC.
            input_lengths = torch.tensor(
                [model.output_length(int(w)) for w in widths],
                dtype=torch.long,
            )

            # Aizsardzība pret kļūdu gadījumā, ja output_length() formula
            # nesakrīt ar faktisko modeļa izvades izmēru.
            if input_lengths.max().item() > T:
                raise RuntimeError(
                    f"input_lengths bigger than T: "
                    f"max(input_lengths)={input_lengths.max().item()}, T={T}"
                )

            # Aprēķinām kļūdu: cik ļoti tīkla prognoze atšķiras no pareizā teksta.
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Trīs obligātie «apmācības rituāla» soļi:
            optimizer.zero_grad()   # 1) atiestatām no iepriekšējās reizes uzkrātās korekcijas
            loss.backward()         # 2) aprēķinām, kā jāmaina katrs svars (kļūdas atpakaļizplatīšana)
            # Aizsardzība pret «eksploziju»: ja korekcijas ir ļoti lielas, ierobežojam to lielumu ar 5.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()        # 3) pielietojam korekcijas - tīkls ir kļuvis nedaudz gudrāks

            step += 1
            # Ik pēc log_every soļiem izdrukājam pašreizējo kļūdu. Tai pakāpeniski jāsamazinās.
            if step % args.log_every == 0:
                print(f"{step}/{args.steps}  failure={loss.item():.4f}")
            # Ik pēc checkpoint_every soļiem (un pašās beigās) saglabājam «kontrolpunktu» -
            # failu ar visiem tīkla svariem. Tas ir kā «saglabāšana spēlē»: vēlāk var
            # to ielādēt un nemācīt tīklu no jauna.
            if step % args.checkpoint_every == 0 or step == args.steps:
                ckpt_path = os.path.join(args.checkpoint_dir, f"crnn_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"saved checkpoint -> {ckpt_path}")
            # Sasniegts nepieciešamais soļu skaits - izejam no pakešu cikla.
            if step >= args.steps:
                break

    print("Complete!")


# Šis bloks izpildās tikai tad, ja fails tiek palaists tieši («python train.py»).
if __name__ == "__main__":
    # ArgumentParser apraksta, kādus iestatījumus var nodot palaišanas laikā.
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument("--steps", type=int, default=2000)         # сколько всего шагов обучения
    parser.add_argument("--batch-size", type=int, default=64)      # сколько примеров в пачке
    parser.add_argument("--lr", type=float, default=1e-3)          # скорость обучения (1e-3 = 0.001)
    parser.add_argument("--synth-per-epoch", type=int, default=5000, help="сколько искусственных примеров делать за одну «эпоху» (один круг обучения)")
    parser.add_argument("--real-data", type=str, default=None, help="путь к папке с настоящими сканами (когда они появятся)")
    parser.add_argument("--num-workers", type=int, default=2)      # параллельные помощники для данных
    parser.add_argument("--log-every", type=int, default=20)       # как часто печатать ошибку
    parser.add_argument("--checkpoint-every", type=int, default=1000)  # как часто сохранять веса
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")  # куда сохранять
    args = parser.parse_args()  # читаем то, что ввёл пользователь
=======
    parser.add_argument("--steps", type=int, default=2000)         # kopējais apmācības soļu skaits
    parser.add_argument("--batch-size", type=int, default=32)      # piemēru skaits paketē
    parser.add_argument("--lr", type=float, default=1e-3)          # mācīšanās ātrums (1e-3 = 0.001)
    parser.add_argument("--synth-per-epoch", type=int, default=5000, help="Cik mākslīgu piemēru vajadzētu izveidot vienā laikmetā (vienā apmācības ciklā)?")
    parser.add_argument("--real-data", type=str, default=None, help="path to folder with real d")
    parser.add_argument("--num-workers", type=int, default=2)      # paralēlie datu sagatavošanas palīgi
    parser.add_argument("--log-every", type=int, default=20)       # cik bieži izdrukāt kļūdu
    parser.add_argument("--checkpoint-every", type=int, default=500)  # cik bieži saglabāt svarus
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")  # kur saglabāt
    args = parser.parse_args()  # nolasām lietotāja ievadītos iestatījumus
>>>>>>> e190ff6678718620cb007d83cf822d4a60364f73

    # Fiksējam «nejaušību» ar skaitli 0, lai palaišanu varētu atkārtot ar tādu pašu rezultātu.
    random.seed(0)
    torch.manual_seed(0)
    train(args)

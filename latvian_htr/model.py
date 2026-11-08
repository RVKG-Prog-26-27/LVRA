"""
Neironu tīkla arhitektūra

Modeli sauc par CRNN: C = konvolucionāls, RNN = rekurents. Šī ir standarta shēma teksta lasīšanai no attēliem.

1) CNN - pārveido attēlu "aprakstu" secībā,
viens apraksts katrai šaurai vertikālai šķēlei
(no kreisās uz labo).

2) BiLSTM - "atmiņas un konteksta izpratne". Nolasa šo secību
un abos virzienos: no kreisās uz labo UN no labās uz kreiso. Pateicoties tam,
tas saprot, ko nozīmē burts blakus citiem burtiem. Piemēram,
nesalasāms līklocis starp "s" un "l" visticamāk ir "a".

3) Lineārs + CTC - "atbilde". Katrai šķēlei tas izvēlas, kurš burts tas ir
(vai "tukš"). CTC ir gudrs veids, kā mācīties, neatzīmējot,
kur tieši attēlā parādās katrs burts.

"""

import torch                
import torch.nn as nn      

class CNNBackbone(nn.Module):
    """
    Izmēru sadalījums:
    B — batch, cik attēlu tiek apstrādāti vienlaicīgi;
    1 — viens kanāls (pelēks, nevis krāsains attēls);
    32 — attēla augstums pikseļos;
    W — attēla platums (katram attēlam atšķirīgs).
    Izvade: augstums ir sarucis līdz 1, bet "kanālu" skaits ir palielinājies līdz 512.
    """
    def __init__(self, x_stride: int = 2):
        super().__init__()
        if x_stride not in (2, 4):
            raise ValueError("x_stride must be 2 or 4")
        # x_stride=4: во 2-м блоке ширина тоже сжимается в 2 раза. Тогда все следующие (самые тяжёлые)
        # свёртки и BiLSTM работают с вдвое меньшей шириной -> обучение примерно в 2 раза быстрее.
        # Обучаемых весов это не меняет (MaxPool без весов), поэтому старые чекпоинты загружаются,
        # но сеть надо дообучить. T = W // 4 - для строк до ~60 символов при высоте 64 этого хватает.
        pool2 = nn.MaxPool2d(2, 2) if x_stride == 4 else nn.MaxPool2d((2, 1), (2, 1))
        # Šeit izmantoto slāņu skaidrojums:
        # - Conv2d - KONVOLŪCIJA aprēķina
        # izvades «kanālu» skaitu (64, 128, 256, 512...).
        # - ReLU - «taisnotājs»: visus negatīvos skaitļus aizstāj ar 0,
        # pozitīvos atstāj. Bez šādiem nelineāriem slāņiem tīkls
        # būtu tikai viena liela reizinātāja un neko sarežģītu
        # nespētu iemācīties.
        # - MaxPool2d(2, 2) - «saspiešana»: no katra 2x2 kvadrāta atstāj
        # tikai lielāko skaitli. Attēls kļūst divreiz mazāks pēc
        # augstuma un platuma, saglabājot svarīgāko informāciju.
        # - MaxPool2d((2, 1), (2, 1)) - tas pats, bet saspiež tikai pēc augstuma
        # (2 reizes), platumu nemainot. Platums mums ir nepieciešams, jo pa to
        # mēs «lasīsim» tekstu no kreisās uz labo pusi.
        # - BatchNorm2d - «normalizācija»: pielāgo skaitļus ērtam
        # mērogam, lai apmācība notiktu ātrāk un stabilāk.
        self.net = nn.Sequential(
            # 1. bloks: 1 kanāls -> 64. Vienīgā vieta, kur saspiežam PLATUMU (2x): 64xW -> 32 x W/2
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            # 2. bloks: 64 -> 128, saspiežam tikai augstumu: -> 16 x W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), pool2,
            # 3. bloks: 128 -> 256 -> 256, tikai augstums: -> 8 x W/2
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            # 4. bloks: 256 -> 512 -> 512 ar normalizāciju, tikai augstums: -> 4 x W/2
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            # Pēdējā konvolūcija ar (4,1) logu: "apēd" atlikušās 4 augstuma rindas -> augstums = 1,
            # platums NEMAINĀS. Tātad T = W // 2 (vecajā versijā bija W-1 un neatbilda output_length!).
            nn.Conv2d(512, 512, (4, 1), 1, 0), nn.ReLU(inplace=True),  # -> 1 x W/2
        )

    def forward(self, x):
        # forward - izpilda slāni, kad tam padod datus.
        return self.net(x)


class BiLSTMHead(nn.Module):
    """
    Tīkla “galva”: nolasa pazīmju secību un rada burtu varbūtības. 

LSTM ir slāņa veids ar "atmiņu". Tas soli pa solim nolasa datus secībā
soli, un atceras to, kas notika iepriekš. Tas ir svarīgi tekstam: saprast
vēstuli, vajag redzēt kaimiņus. "Bi" (divvirzienu) nozīmē "divvirzienu":
divi LSTM, viens nolasa no kreisās puses uz labo, otrs no labās uz kreiso, un rezultāti
turēties kopā. Tātad katra pozīcija zina, kas notika pirms tās un kas notiks pēc tam.
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        # in_dim - cik skaitļi apraksta vienu attēla daļu (mums 512);
        # hidden - LSTM «atmiņas» izmērs (cik skaitļus tas saglabā);
        # num_layers=2 - divi viens virs otra novietoti LSTM slāņi: otrais lasa
        # pirmā rezultātu un apstrādā to dziļāk;
        # bidirectional=True - ieslēdzam lasīšanu abos virzienos;
        # batch_first=False - datu dimensiju secība: [solis, attēls paketē, pazīmes].
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=num_layers, bidirectional=True, batch_first=False
        )
        # Linear - parasts «pilnībā savienots» slānis: tas paņem LSTM skaitļus un pārveido
        # tos par katra alfabēta simbola novērtējumiem. Ievade ir hidden * 2, jo
        # abi virzieni tiek apvienoti (katrs dod hidden skaitļus).
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        # x: [T, B, in_dim]
        # T - cik attēla «daļu» ir horizontāli (laika soļu),
        # B - cik attēlu ir paketē,
        # in_dim - cik pazīmju apraksta vienu daļu.
        out, _ = self.lstm(x)   # out - tas, ko LSTM «saprata» katrā solī ("_" - atmiņa, tā mums nav vajadzīga)
        return self.fc(out)     # [T, B, num_classes]: katra simbola novērtējums katram solim


class CRNN(nn.Module):
    """Viss modelis: "acis" (CNN) + "izpratne" (BiLSTM)."""

    def __init__(self, num_classes: int, lstm_hidden: int = 256, x_stride: int = 2):
        super().__init__()
        self.x_stride = x_stride
        self.cnn = CNNBackbone(x_stride)  # acis
        # Galva: ievadē saņem 512 pazīmes (tik izvades kanālu ir CNN),
        # izvadē - num_classes novērtējumus (pēc alfabēta simbolu skaita, ieskaitot blank).
        self.rnn = BiLSTMHead(in_dim=512, hidden=lstm_hidden, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 1, 64, W] — attēlu virkņu partija.
        Atgriež varbūtību logaritmus formā [T, B, klašu_skaits],
        kur T ir secības garums pēc tam, kad CNN ir saspiedis attēla platumu.
        """
        feats = self.cnn(images)              # [B, 512, 1, W']  - CNN atrada pazīmes
        feats = feats.squeeze(2)              # [B, 512, W']     - noņemam lieko augstuma dimensiju (tā ir 1)
        # permute pārkārto dimensijas: bija [attēls, pazīmes, solis],
        # nepieciešams [solis, attēls, pazīmes] - tieši šādus datus sagaida LSTM.
        feats = feats.permute(2, 0, 1)        # [W'(=T), B, 512]
        logits = self.rnn(feats)              # [T, B, num_classes]  - simbolu novērtējumi
        # log_softmax pārvērš novērtējumus par varbūtību logaritmiem (summa ir 100% pa
        # simboliem katrā solī). Tieši šādā formā tos sagaida CTC zaudējumu funkcija.
        # .float(): при обучении в mixed precision softmax и CTC всё равно считаем в float32 (точность).
        return logits.float().log_softmax(dim=2)

    def output_length(self, input_width: int) -> int:
        """Cik soļus T tiks iegūts CNN izvadē attēlam ar noteiktu platumu (nepieciešams CTC)?
        x_stride=2: platums tiek saspiests vienu reizi (pirmais MaxPool(2,2)) -> T = W // 2.
        x_stride=4: divreiz -> T = (W // 2) // 2.
        Ja maini CNN (pooling/kernel), JĀLABO ŠĪ FORMULA, citādi CTC redzēs nepareizu garumu."""
        t = input_width // 2
        if self.x_stride == 4:
            t //= 2
        return max(t, 1)

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        """То же самое сразу для всего батча (widths - long тензор [B])."""
        t = widths // 2
        if self.x_stride == 4:
            t = t // 2
        return t.clamp(min=1)


def save_checkpoint(model: CRNN, path: str, step: int | None = None):
    """Чекпоинт = веса + настройки архитектуры (x_stride), чтобы infer/evaluate собрали такую же сеть."""
    torch.save({"model": model.state_dict(), "x_stride": model.x_stride, "step": step}, path)


def load_checkpoint(path: str, map_location="cpu") -> tuple[dict, int]:
    """Читает и новый формат (dict с 'model'), и старый (просто state_dict). Возвращает (веса, x_stride)."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("x_stride", 2)
    return ckpt, 2


def load_model(path: str, num_classes: int, device) -> CRNN:
    """Создаёт CRNN с той же архитектурой, что в чекпоинте, и загружает веса."""
    state, x_stride = load_checkpoint(path, device)
    model = CRNN(num_classes=num_classes, x_stride=x_stride).to(device)
    model.load_state_dict(state)
    return model

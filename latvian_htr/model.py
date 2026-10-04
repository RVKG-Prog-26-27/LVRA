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
    def __init__(self):
        super().__init__()
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
            # 1. bloks: 1 kanāls -> 64 kanāli. Pēc tam saspiešana divreiz.
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),      # 32x W -> 16 x W/2
            # 2. bloks: 64 -> 128 kanāli. Atkal saspiešana divreiz.
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),    # -> 8 x W/4
            # 3. bloks: divas konvolūcijas pēc kārtas (128 -> 256 -> 256), pēc tam saspiešana TIKAI pēc augstuma.
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),  # -> 4 x W/4
            # 4. bloks: 256 -> 512 -> 512 kanāli ar normalizāciju, atkal saspiešana tikai pēc augstuma.
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),  # -> 2 x W/4
            # Pēdējā konvolūcija ar 2x2 logu bez atkāpes: «apēd» atlikušās 2 augstuma rindas
            # vienā. Tagad augstums = 1, bet platums ir par 1 mazāks.
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(inplace=True),  # -> 1 x (W/4 - 1)
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

    def __init__(self, num_classes: int, lstm_hidden: int = 256):
        super().__init__()
        self.cnn = CNNBackbone()  # acis
        # Galva: ievadē saņem 512 pazīmes (tik izvades kanālu ir CNN),
        # izvadē - num_classes novērtējumus (pēc alfabēta simbolu skaita, ieskaitot blank).
        self.rnn = BiLSTMHead(in_dim=512, hidden=lstm_hidden, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 1, 32, W] — attēlu virkņu partija.
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
        return logits.log_softmax(dim=2)

    def output_length(self, input_width: int) -> int:
        """Cik soļus T tiks iegūts CNN izvadē attēlam ar noteiktu platumu (nepieciešams CTC)?"""
        w = input_width // 2 // 2  # divas MaxPool(2,2) saspiešanas samazināja platumu četras reizes
        w = w - 1                  # pēdējā konvolūcija ar 2 izmēra logu noņēma vēl vienu kolonnu
        # max(w, 1) - garantējam, ka rezultāts nav mazāks par 1 (arī ļoti šauram attēlam).
        return max(w, 1)

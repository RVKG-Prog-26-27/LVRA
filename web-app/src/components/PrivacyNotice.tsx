import { useEffect, useRef } from 'react';
import { config } from '../config/config';
import { lv } from '../i18n/lv';

interface Props {
  open: boolean;
  onClose: () => void;
}

/**
 * Privacy notice in Latvian. The contact email comes from src/config/config.ts.
 */
export function PrivacyNotice({ open, onClose }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  const email = config.privacyContactEmail;

  return (
    <dialog ref={dialogRef} className="privacy" aria-labelledby="privacy-title" onClose={onClose}>
      <h2 id="privacy-title">{lv.privacyLink}</h2>
      <p className="effective">Spēkā no 2026. gada 8. novembra</p>

      <h3>Kas apstrādā datus</h3>
      <p>
        LVRA lietotni izstrādā un uztur komanda Slepie, kas atrodas Latvijā. Saziņai:{' '}
        <a href={`mailto:${email}`}>{email}</a>.
      </p>

      <h3>Kādi dati tiek apstrādāti</h3>
      <p>
        Attēls, ko jūs izvēlaties vai nofotografējat, no tā izgrieztie teksta rindu attēli un atpazītais teksts. Mēs
        neprasām vārdu, e-pastu vai citu informāciju par jums. Ja attēlā ir rakstīti personas dati, piemēram, vārdi vai
        adreses, arī tie tiek apstrādāti.
      </p>

      <h3>Kā notiek apstrāde</h3>
      <p>
        Jūsu ierīcē attēls tiek samazināts un sadalīts atsevišķos teksta rindu attēlos. Viss attēls no jūsu ierīces
        netiek nosūtīts. Rindu attēli kopā ar tabulu (CSV) tiek nosūtīti atpazīšanas serverim Eiropas Savienībā.
        Serveris atgriež atpazīto tekstu, un lietotne to parāda redaktorā. Jūsu labojumi redaktorā netiek nosūtīti.
      </p>

      <h3>Kāpēc un uz kāda pamata</h3>
      <p>
        Teksta atpazīšana: lai sniegtu pakalpojumu, ko pieprasāt (Vispārīgās datu aizsardzības regulas 6. panta
        1. punkta b) apakšpunkts).
      </p>
      <p>AI modeļa apmācība: tikai ar jūsu piekrišanu (6. panta 1. punkta a) apakšpunkts).</p>

      <h3>Kas var redzēt datus un kur tie atrodas</h3>
      <p>
        Datus redz tikai komandas Slepie dalībnieki. Ja esat devis piekrišanu apmācībai, viņi apskata rindu attēlus un
        atpazīto tekstu, lai atrastu un izlabotu AI kļūdas. Dati tiek glabāti PythonAnywhere serveros Eiropas
        Savienībā un netiek nodoti citām personām.
      </p>

      <h3>Cik ilgi dati tiek glabāti</h3>
      <p>
        Bez piekrišanas apmācībai rindu attēli un tabula tiek dzēsti uzreiz pēc teksta atpazīšanas. Ar piekrišanu tie
        tiek glabāti līdz LVRA projekta beigām un pēc tam dzēsti. Atpazītais teksts jūsu pārlūkā pazūd, kad aizverat
        lapu vai sākat no jauna.
      </p>

      <h3>Kas tiek saglabāts jūsu ierīcē</h3>
      <p>
        Lietotne jūsu pārlūkā saglabā tikai to, ka esat pieņēmis lietošanas noteikumus, un jūsu izvēli par apmācību,
        lai logs neparādītos katrā apmeklējumā. Lietotne neizmanto sīkdatnes, reklāmas vai lietošanas analīzes rīkus.
      </p>

      <h3>Piekrišanas atsaukšana</h3>
      <p>
        Izvēli varat mainīt jebkurā laikā, lapas apakšā atverot "{lv.termsLink}". Tā attiecas uz turpmāk
        augšupielādētajiem attēliem. Jau nosūtītos rindu attēlus mēs nevaram atrast un dzēst, jo tie nav saistīti ar
        datiem, pēc kuriem jūs varētu identificēt.
      </p>

      <h3>Jūsu tiesības</h3>
      <p>
        Saskaņā ar Vispārīgo datu aizsardzības regulu jums ir tiesības saņemt informāciju par savu datu apstrādi. Ar
        jautājumiem rakstiet uz <a href={`mailto:${email}`}>{email}</a>. Jums ir arī tiesības iesniegt sūdzību Datu
        valsts inspekcijā (
        <a href="https://www.dvi.gov.lv" target="_blank" rel="noreferrer">
          www.dvi.gov.lv
        </a>
        ).
      </p>

      <div className="actions">
        <button type="button" className="btn btn-primary" onClick={onClose} autoFocus>
          {lv.close}
        </button>
      </div>
    </dialog>
  );
}

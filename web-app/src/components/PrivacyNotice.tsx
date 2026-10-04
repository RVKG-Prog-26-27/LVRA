import { useEffect, useRef } from 'react';
import { config } from '../config/config';
import { lv } from '../i18n/lv';

interface Props {
  open: boolean;
  onClose: () => void;
}

/**
 * Privacy notice in Latvian. The contact email comes from configuration
 * (VITE_PRIVACY_CONTACT_EMAIL) and is a placeholder until set.
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

      <h3>Kas apstrādā datus</h3>
      <p>LVRA lietotni izstrādā un uztur komanda Slepie, kas atrodas Latvijā.</p>

      <h3>Kādi dati tiek apstrādāti</h3>
      <p>Tikai attēls, ko jūs izvēlaties vai nofotografējat, un no tā atpazītais teksts. Mēs neprasām vārdu, e-pastu vai citu informāciju par jums.</p>

      <h3>Kāpēc</h3>
      <p>Lai atpazītu attēlā redzamo rokrakstu un parādītu to jums kā digitālu tekstu.</p>

      <h3>Kā notiek apstrāde</h3>
      <p>Jūsu ierīcē attēls tiek samazināts un pēc tam nosūtīts atpazīšanas serverim. Serveris atgriež atpazīto tekstu.</p>

      <h3>Cik ilgi dati tiek glabāti</h3>
      <p>Attēls netiek saglabāts. Tas tiek dzēsts uzreiz pēc teksta atpazīšanas gan lietotnē, gan serverī. Atpazītais teksts atrodas tikai jūsu pārlūkā un pazūd, kad aizverat lapu vai sākat no jauna.</p>

      <h3>Sīkdatnes un izsekošana</h3>
      <p>Lietotne neizmanto sīkdatnes, reklāmas vai lietošanas analīzes rīkus.</p>

      <h3>Jūsu tiesības</h3>
      <p>
        Saskaņā ar Vispārīgo datu aizsardzības regulu jums ir tiesības saņemt informāciju par savu datu apstrādi. Ar jautājumiem rakstiet uz{' '}
        <a href={`mailto:${email}`}>{email}</a>. Jums ir arī tiesības iesniegt sūdzību Datu valsts inspekcijā (
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

import { useEffect, useRef } from 'react';
import { config } from '../config/config';
import { lv } from '../i18n/lv';

interface Props {
  open: boolean;
  /**
   * true on the first visit (or after the terms changed): the window can only
   * be left by choosing one of the two buttons.
   */
  required: boolean;
  /** The current choice, shown when the window is reopened from the footer. */
  currentConsent: boolean | null;
  onChoose: (trainingConsent: boolean) => void;
  onClose: () => void;
  onOpenPrivacy: () => void;
}

/**
 * Terms of use (lietošanas noteikumi) in Latvian. Opens by itself on the first
 * visit. The training consent is optional: the two buttons have equal weight
 * and nothing is pre-selected.
 */
export function TermsDialog({ open, required, currentConsent, onChoose, onClose, onOpenPrivacy }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  // While a choice is required, Escape must not close the window.
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    const onCancel = (event: Event) => {
      if (required) event.preventDefault();
    };
    dialog.addEventListener('cancel', onCancel);
    return () => dialog.removeEventListener('cancel', onCancel);
  }, [required]);

  const email = config.privacyContactEmail;

  return (
    <dialog
      ref={dialogRef}
      className="privacy terms"
      aria-labelledby="terms-title"
      onClose={() => {
        if (!required) onClose();
      }}
    >
      <h2 id="terms-title" tabIndex={-1} autoFocus>
        {lv.termsLink}
      </h2>
      <p className="effective">Spēkā no 2026. gada 8. novembra</p>

      <h3>1. Kas nodrošina lietotni</h3>
      <p>
        LVRA (Latvijas Valodas Rokraksta Atpazīšana) lietotni izstrādā un uztur komanda Slepie, kas atrodas Latvijā.
        Saziņai: <a href={`mailto:${email}`}>{email}</a>.
      </p>

      <h3>2. Izstrādes versija</h3>
      <p>
        LVRA vēl tiek izstrādāta. Lietotne var atpazīt tekstu kļūdaini vai uz laiku nedarboties, tāpēc mēs
        negarantējam, ka atpazītais teksts būs precīzs. Pirms teksta izmantošanas, lūdzu, pārbaudiet to.
      </p>

      <h3>3. Kas var lietot lietotni</h3>
      <p>Lietotni drīkst izmantot personas, kuras ir vismaz 13 gadus vecas.</p>

      <h3>4. Kādus attēlus augšupielādēt</h3>
      <p>
        Augšupielādējiet tikai tādus attēlus, kurus jums ir tiesības izmantot. Neaugšupielādējiet attēlus, kuros ir
        sensitīvi dati par jums vai citiem cilvēkiem, piemēram, informācija par veselību, personas kodi vai dokumentu
        numuri.
      </p>

      <h3>5. Piekrišana AI modeļa apmācībai</h3>
      <p>
        Šī sadaļa nav obligāta. Ja izvēlaties «{lv.termsAcceptMandatory}», rindu attēli tiek dzēsti uzreiz pēc teksta
        atpazīšanas.
      </p>
      <p>
        Ja izvēlaties «{lv.termsAcceptAll}», jūsu attēlā atrastās teksta rindas (PNG attēli) un atpazītais teksts
        tiek saglabāti un izmantoti LVRA rokraksta atpazīšanas modeļa uzlabošanai. Komandas Slepie dalībnieki tos
        apskata, lai atrastu un izlabotu AI kļūdas.
      </p>
      <p>
        Izvēli varat mainīt jebkurā laikā, lapas apakšā atverot "{lv.termsLink}". Jaunā izvēle attiecas uz attēliem,
        ko augšupielādēsiet pēc tam. Jau nosūtītos rindu attēlus mēs nevaram atrast un dzēst, jo tie nav saistīti ne ar
        jūsu vārdu, ne ar citiem datiem, pēc kuriem jūs varētu identificēt.
      </p>

      <h3>6. Izmaiņas noteikumos</h3>
      <p>
        Mēs varam šos noteikumus mainīt. Ja tie mainās, lietotne tos parādīs vēlreiz, un jums būs jāpiekrīt jaunajai
        versijai.
      </p>

      <h3>7. Saziņa</h3>
      <p>
        Jautājumus rakstiet uz <a href={`mailto:${email}`}>{email}</a>. Kā mēs apstrādājam datus, aprakstīts{' '}
        <button type="button" className="link" onClick={onOpenPrivacy}>
          Privātuma paziņojumā
        </button>
        .
      </p>

      {currentConsent !== null && (
        <p className="terms-current" role="status">
          {currentConsent ? lv.termsCurrentConsent : lv.termsCurrentNoConsent}
        </p>
      )}

      <div className="actions terms-actions">
        <button type="button" className="btn btn-primary" onClick={() => onChoose(true)}>
          {lv.termsAcceptAll}
        </button>
        <button type="button" className="btn btn-primary" onClick={() => onChoose(false)}>
          {lv.termsAcceptMandatory}
        </button>
        {!required && (
          <button type="button" className="btn btn-quiet" onClick={onClose}>
            {lv.close}
          </button>
        )}
      </div>
    </dialog>
  );
}

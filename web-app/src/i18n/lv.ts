/**
 * All user-facing texts in Latvian. Keeping them in one file makes wording
 * reviews easy and keeps technical terms out of components.
 */
import type { AppErrorCode } from '../recognition/errors';

export const lv = {
  appName: 'LVRA',
  appFullName: 'Latvijas Valodas Rokraksta Atpazīšana',
  intro:
    'Nofotografējiet vai izvēlieties attēlu ar rokrakstu latviešu valodā, un LVRA to pārvērtīs tekstā, ko varat labot un kopēt.',

  chooseImage: 'Izvēlēties attēlu',
  takePhoto: 'Uzņemt foto',
  formatHint: 'Izmantojiet JPEG vai PNG attēlu.',
  privacyShort: 'Attēls tiek izmantots tikai teksta atpazīšanai un pēc tam tiek dzēsts.',
  loadingImage: 'Atver attēlu…',
  selectionCancelled:
    'Attēls netika izvēlēts. Ja kamera vai attēlu galerija neatveras, pārlūka vai ierīces iestatījumos atļaujiet tai piekļuvi.',

  previewTitle: 'Vai šis ir pareizais attēls?',
  previewHint: 'Ja teksts ir šķībs vai apgriezts, pagrieziet attēlu.',
  previewAlt: 'Izvēlētā attēla priekšskatījums',
  rotateLeft: 'Pagriezt pa kreisi',
  rotateRight: 'Pagriezt pa labi',
  chooseAnother: 'Izvēlēties citu attēlu',
  recognize: 'Atpazīt tekstu',

  processing: 'Apstrādā attēlu…',
  processingHint: 'Tas var aizņemt līdz vienai minūtei.',
  cancel: 'Atcelt',

  resultTitle: 'Atpazītais teksts',
  resultHint: 'Pārbaudiet tekstu un, ja nepieciešams, izlabojiet to.',
  uncertainLegend:
    'Iekrāsotie vārdi, iespējams, ir atpazīti neprecīzi. Kad vārdu izlabojat, iekrāsojums pazūd.',
  editorLabel: 'Atpazītais teksts, ko var labot',
  copy: 'Kopēt tekstu',
  copied: 'Teksts nokopēts!',
  copyFailed: 'Neizdevās nokopēt tekstu. Iezīmējiet to un nokopējiet manuāli.',
  emptyText: 'Teksta lauks ir tukšs.',
  startOver: 'Atpazīt citu attēlu',

  errorTitle: 'Neizdevās atpazīt tekstu',
  retry: 'Mēģināt vēlreiz',

  demoBanner:
    'Demonstrācijas režīms: rokraksta atpazīšana vēl tiek izstrādāta, tāpēc tiek rādīts parauga teksts, nevis jūsu attēla saturs.',

  footer: 'LVRA · Izstrādāja Slepie',
  privacyLink: 'Privātuma paziņojums',
  close: 'Aizvērt',

  errors: {
    NO_IMAGE: 'Vispirms izvēlieties vai nofotografējiet attēlu.',
    NOT_AN_IMAGE: 'Izvēlētais fails nav attēls. Lūdzu, izvēlieties JPEG vai PNG attēlu.',
    UNSUPPORTED_FORMAT: 'Šo attēlu nevar atvērt. Lūdzu, izvēlieties JPEG vai PNG attēlu.',
    IMAGE_PROCESSING_FAILED: 'Attēlu neizdevās sagatavot. Mēģiniet izvēlēties citu attēlu.',
    NO_TEXT_FOUND:
      'Attēlā netika atrasts rokraksts. Pārliecinieties, ka teksts ir labi redzams, un mēģiniet vēlreiz.',
    POOR_QUALITY:
      'Attēls ir pārāk neskaidrs vai tumšs. Nofotografējiet tekstu tuvāk un labākā apgaismojumā.',
    INVALID_IMAGE: 'Šo attēlu nevar apstrādāt. Mēģiniet izvēlēties citu attēlu.',
    SERVICE_UNAVAILABLE: 'Atpazīšana pašlaik nav pieejama. Lūdzu, mēģiniet vēlāk.',
    TIMEOUT: 'Atpazīšana aizņem pārāk ilgi. Lūdzu, mēģiniet vēlreiz.',
    OFFLINE: 'Nav interneta savienojuma. Pārbaudiet savienojumu un mēģiniet vēlreiz.',
    CANCELLED: 'Atpazīšana tika atcelta.',
    UNKNOWN: 'Radās neparedzēta kļūda. Lūdzu, mēģiniet vēlreiz.',
  } satisfies Record<AppErrorCode, string>,
};

export function errorMessage(code: AppErrorCode): string {
  return lv.errors[code] ?? lv.errors.UNKNOWN;
}

<<<<<<< HEAD
import { useEffect, useMemo, useState } from 'react';
=======
//builds every screen of the app itself

import { useMemo, useState } from 'react';
>>>>>>> origin/web-app
import { useLvraApp } from './app/useLvraApp';
import { loadConsent, saveConsent, type ConsentChoice } from './consent/consentStorage';
import { DevPanel } from './components/DevPanel';
import { ErrorMessage } from './components/ErrorMessage';
import { RotateLeftIcon, RotateRightIcon } from './components/Icons';
import { ImagePicker } from './components/ImagePicker';
import { ImagePreview } from './components/ImagePreview';
import { PrivacyNotice } from './components/PrivacyNotice';
import { ProcessingStatus } from './components/ProcessingStatus';
import { ResultEditor } from './components/ResultEditor';
import { TermsDialog } from './components/TermsDialog';
import { lv } from './i18n/lv';
import { devInspector } from './recognition/devInspector';
import { mockSettings, type Backend } from './recognition/mockSettings';

export default function App() {
  // The terms window opens by itself until the user has chosen (and again when the terms version changes).
  const [consent, setConsent] = useState<ConsentChoice | null>(() => loadConsent());
  const [termsOpen, setTermsOpen] = useState(consent === null);
  const [privacyOpen, setPrivacyOpen] = useState(false);
  const trainingConsent = consent?.trainingConsent ?? false;
  const { state, selectFile, rotate, recognize, cancel, reset } = useLvraApp(trainingConsent);

  const showDevPanel = useMemo(() => new URLSearchParams(window.location.search).get('dev') === '1', []);
  const [backend, setBackend] = useState<Backend>(mockSettings.backend);
  useEffect(() => {
    if (showDevPanel) devInspector.enable();
  }, [showDevPanel]);

  const chooseConsent = (value: boolean) => {
    const choice = { trainingConsent: value };
    saveConsent(choice);
    setConsent(choice);
    setTermsOpen(false);
  };

  //static - doesn't depend on the current state of the screen
  return (
    <div className="page">
      <header className="masthead">
        <h1>
          <span className="wordmark">{lv.appName}</span>
          <span className="fullname">{lv.appFullName}</span>
        </h1>
        <p className="intro">{lv.intro}</p>
      </header>

      <p className="demo-banner" role="note">
        {backend === 'mock' ? lv.demoBanner : lv.aiBanner}
      </p>

      <main className="sheet">
        {state.step === 'pick' && (
          <>
            <ImagePicker onFile={selectFile} />
            <p className="muted">{lv.formatHint}</p>
            {consent && (
              <p className="muted">{consent.trainingConsent ? lv.privacyShortConsent : lv.privacyShortNoConsent}</p>
            )}
            {state.error && <ErrorMessage code={state.error} />}
            {state.notice === 'cancelled' && (
              <p className="notice" role="status">
                {lv.selectionCancelled}
              </p>
            )}
          </>
        )}

        // loading screen - short message
        {state.step === 'loading' && (
          <p className="muted" role="status" aria-live="polite">
            {lv.loadingImage}
          </p>
        )}


        //preview and recognition screen - progress state and "Atcelt"
        {(state.step === 'preview' || state.step === 'recognizing') && (
          <>
            <h2>{lv.previewTitle}</h2>
            <div className="preview-wrap">
              <ImagePreview image={state.image} rotation={state.rotation} busy={state.step === 'recognizing'} />
              {state.step === 'recognizing' && <ProcessingStatus onCancel={cancel} />}
            </div>
            

            //preview screen -- 2 buttons
            {state.step === 'preview' && (
              <>
                <p className="muted">{lv.previewHint}</p>
                <div className="actions">
                  <button type="button" className="btn btn-quiet" onClick={() => rotate(-90)}>
                    <RotateLeftIcon />
                    {lv.rotateLeft}
                  </button>
                  <button type="button" className="btn btn-quiet" onClick={() => rotate(90)}>
                    <RotateRightIcon />
                    {lv.rotateRight}
                  </button>
                </div>


                //errors - displays all error types
                {state.error ? (
                  <ErrorMessage code={state.error} withTitle>
                    <button type="button" className="btn btn-primary" onClick={recognize}>
                      {lv.retry}
                    </button>
                    <button type="button" className="btn btn-quiet" onClick={reset}>
                      {lv.chooseAnother}
                    </button>
                  </ErrorMessage>
                ) : (
                  <div className="actions actions-main">
                    <button type="button" className="btn btn-primary" onClick={recognize}>
                      {lv.recognize}
                    </button>
                    <button type="button" className="btn btn-quiet" onClick={reset}>
                      {lv.chooseAnother}
                    </button>
                  </div>
                )}
              </>
            )}
          </>
        )}


        //result screen - text aditor
        {state.step === 'result' && <ResultEditor segments={state.segments} onStartOver={reset} />}
      </main>

      <footer className="footer">
        <span>{lv.footer}</span>
        <span className="footer-links">
          <button type="button" className="link" onClick={() => setTermsOpen(true)}>
            {lv.termsLink}
          </button>
          <button type="button" className="link" onClick={() => setPrivacyOpen(true)}>
            {lv.privacyLink}
          </button>
        </span>
      </footer>

      {showDevPanel && <DevPanel backend={backend} onBackendChange={setBackend} />}
      <TermsDialog
        open={termsOpen}
        required={consent === null}
        currentConsent={consent ? consent.trainingConsent : null}
        onChoose={chooseConsent}
        onClose={() => setTermsOpen(false)}
        onOpenPrivacy={() => setPrivacyOpen(true)}
      />
      <PrivacyNotice open={privacyOpen} onClose={() => setPrivacyOpen(false)} />
    </div>
  );
}

import { useEffect, useMemo, useState } from 'react';
import { useLvraApp } from './app/useLvraApp';
import { loadConsent, saveConsent, type ConsentChoice } from './consent/consentStorage';
import { DevPanel } from './components/DevPanel';
import { ErrorMessage } from './components/ErrorMessage';
import { ImagePicker } from './components/ImagePicker';
import { PhotoList } from './components/PhotoList';
import { PrivacyNotice } from './components/PrivacyNotice';
import { ProcessingStatus } from './components/ProcessingStatus';
import { ResultTabs } from './components/ResultTabs';
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
  const { state, selectFiles, rotate, remove, recognize, cancel, reset } = useLvraApp(trainingConsent);

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
            <ImagePicker onFiles={selectFiles} />
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
            {state.notice === 'sizeLimit' && (
              <p className="notice" role="status">
                {lv.sizeLimit}
              </p>
            )}
          </>
        )}

        {(state.step === 'loading' || state.step === 'preview' || state.step === 'recognizing') &&
          state.items.length > 0 && (
            <>
              <h2>{state.items.length > 1 ? lv.previewTitleMany : lv.previewTitle}</h2>
              <PhotoList
                items={state.items}
                busy={state.step !== 'preview'}
                onRotate={rotate}
                onRemove={remove}
              />
            </>
          )}

        {state.step === 'loading' && (
          <p className="muted" role="status" aria-live="polite">
            {lv.loadingImage}
          </p>
        )}

        {state.step === 'recognizing' && (
          <ProcessingStatus onCancel={cancel} current={state.current + 1} total={state.items.length} />
        )}

        {state.step === 'preview' && (
          <>
            <p className="muted">{lv.previewHint}</p>
            {state.loadError && <ErrorMessage code={state.loadError} />}
            {state.notice === 'sizeLimit' && (
              <p className="notice" role="status">
                {lv.sizeLimit}
              </p>
            )}
            <ImagePicker onFiles={selectFiles} variant="add" />

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

        {state.step === 'result' && <ResultTabs results={state.results} onStartOver={reset} />}
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

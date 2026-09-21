import { useMemo, useState } from 'react';
import { useLvraApp } from './app/useLvraApp';
import { DevPanel } from './components/DevPanel';
import { ErrorMessage } from './components/ErrorMessage';
import { RotateLeftIcon, RotateRightIcon } from './components/Icons';
import { ImagePicker } from './components/ImagePicker';
import { ImagePreview } from './components/ImagePreview';
import { PrivacyNotice } from './components/PrivacyNotice';
import { ProcessingStatus } from './components/ProcessingStatus';
import { ResultEditor } from './components/ResultEditor';
import { lv } from './i18n/lv';
import { isMockMode } from './recognition/createRecognitionService';

export default function App() {
  const { state, selectFile, rotate, recognize, cancel, reset } = useLvraApp();
  const [privacyOpen, setPrivacyOpen] = useState(false);
  const showDevPanel = useMemo(() => new URLSearchParams(window.location.search).get('dev') === '1', []);

  return (
    <div className="page">
      <header className="masthead">
        <h1>
          <span className="wordmark">{lv.appName}</span>
          <span className="fullname">{lv.appFullName}</span>
        </h1>
        <p className="intro">{lv.intro}</p>
      </header>

      {isMockMode && (
        <p className="demo-banner" role="note">
          {lv.demoBanner}
        </p>
      )}

      <main className="sheet">
        {state.step === 'pick' && (
          <>
            <ImagePicker onFile={selectFile} />
            <p className="muted">{lv.formatHint}</p>
            <p className="muted">{lv.privacyShort}</p>
            {state.error && <ErrorMessage code={state.error} />}
            {state.notice === 'cancelled' && (
              <p className="notice" role="status">
                {lv.selectionCancelled}
              </p>
            )}
          </>
        )}

        {state.step === 'loading' && (
          <p className="muted" role="status" aria-live="polite">
            {lv.loadingImage}
          </p>
        )}

        {(state.step === 'preview' || state.step === 'recognizing') && (
          <>
            <h2>{lv.previewTitle}</h2>
            <div className="preview-wrap">
              <ImagePreview image={state.image} rotation={state.rotation} busy={state.step === 'recognizing'} />
              {state.step === 'recognizing' && <ProcessingStatus onCancel={cancel} />}
            </div>

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

        {state.step === 'result' && <ResultEditor segments={state.segments} onStartOver={reset} />}
      </main>

      <footer className="footer">
        <span>{lv.footer}</span>
        <button type="button" className="link" onClick={() => setPrivacyOpen(true)}>
          {lv.privacyLink}
        </button>
      </footer>

      {showDevPanel && <DevPanel />}
      <PrivacyNotice open={privacyOpen} onClose={() => setPrivacyOpen(false)} />
    </div>
  );
}

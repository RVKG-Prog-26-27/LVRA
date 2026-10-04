/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_RECOGNITION_MODE?: string;
  readonly VITE_RECOGNITION_API_URL?: string;
  readonly VITE_PRIVACY_CONTACT_EMAIL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

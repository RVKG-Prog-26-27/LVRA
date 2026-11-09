import type { ReactNode } from 'react';
import { errorMessage, lv } from '../i18n/lv';
import type { AppErrorCode } from '../recognition/errors';

interface Props {
  code: AppErrorCode;
  /** Show the "Neizdevās atpazīt tekstu" heading (recognition errors). */
  withTitle?: boolean;
  children?: ReactNode;
}

export function ErrorMessage({ code, withTitle, children }: Props) {
  return (
    <div className="error" role="alert">
      {withTitle && <p className="error-title">{lv.errorTitle}</p>}
      <p>{errorMessage(code)}</p>
      {children && <div className="actions">{children}</div>}
    </div>
  );
}

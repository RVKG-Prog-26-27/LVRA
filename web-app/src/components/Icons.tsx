/** Small inline icons (no external icon requests). Decorative, hidden from screen readers. */
const common = {
  width: 20,
  height: 20,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

export const ImageIcon = () => (
  <svg {...common}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <circle cx="9" cy="10" r="1.8" />
    <path d="M21 16l-5-5-8 9" />
  </svg>
);

export const CameraIcon = () => (
  <svg {...common}>
    <path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z" />
    <circle cx="12" cy="13.5" r="3.5" />
  </svg>
);

export const RotateLeftIcon = () => (
  <svg {...common}>
    <path d="M4 5v5h5" />
    <path d="M5.5 15a7 7 0 1 0 1.2-7.4L4 10" />
  </svg>
);

export const RotateRightIcon = () => (
  <svg {...common}>
    <path d="M20 5v5h-5" />
    <path d="M18.5 15a7 7 0 1 1-1.2-7.4L20 10" />
  </svg>
);

export const CopyIcon = () => (
  <svg {...common}>
    <rect x="8" y="8" width="12" height="12" rx="2" />
    <path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" />
  </svg>
);

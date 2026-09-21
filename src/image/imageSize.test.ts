import { fitWithin, rotateBy, rotatedSize } from './imageSize';

describe('fitWithin (1080p limit)', () => {
  it('shrinks a large landscape photo to fit 1920 x 1080', () => {
    expect(fitWithin({ width: 4000, height: 3000 }, 1920, 1080)).toEqual({ width: 1440, height: 1080 });
  });

  it('shrinks a large portrait photo to fit 1080 x 1920', () => {
    expect(fitWithin({ width: 3000, height: 4000 }, 1920, 1080)).toEqual({ width: 1080, height: 1440 });
  });

  it('limits very wide images by the long side', () => {
    expect(fitWithin({ width: 6000, height: 1000 }, 1920, 1080)).toEqual({ width: 1920, height: 320 });
  });

  it('never enlarges small images', () => {
    expect(fitWithin({ width: 800, height: 600 }, 1920, 1080)).toEqual({ width: 800, height: 600 });
  });
});

describe('rotation', () => {
  it('cycles through quarter turns in both directions', () => {
    expect(rotateBy(0, 90)).toBe(90);
    expect(rotateBy(270, 90)).toBe(0);
    expect(rotateBy(0, -90)).toBe(270);
  });

  it('swaps width and height for 90 and 270 degrees', () => {
    expect(rotatedSize({ width: 400, height: 300 }, 90)).toEqual({ width: 300, height: 400 });
    expect(rotatedSize({ width: 400, height: 300 }, 180)).toEqual({ width: 400, height: 300 });
  });
});
